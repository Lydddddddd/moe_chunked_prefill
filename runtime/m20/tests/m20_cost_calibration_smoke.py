#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


PREDICTOR_DIR = Path(__file__).resolve().parents[1] / "inter_layer_predictor"
sys.path.insert(0, str(PREDICTOR_DIR))

from calibrate_m20_b_cost import (  # noqa: E402
    OracleCoverageIndex,
    batch_request_id,
    calibration_gate,
    fit_compute_model,
    fit_transport_model,
    parse_action_timing,
)


def test_oracle_coverage_recomputes_frequency_hits() -> None:
    import numpy as np

    routed = np.asarray(
        [
            [[1, 2], [3, 4]],
            [[1, 3], [4, 5]],
        ],
        dtype=np.int32,
    )
    index = OracleCoverageIndex(
        {128: routed}, prompt_indices=[128], route_top_k=2
    )
    request_id = batch_request_id(128, 1, 0, 2)
    covered = index.covered_routes(
        {
            "request_ids": [request_id],
            "chunk_indices": [1],
            "token_spans": [[0, 2]],
            "token_count": 2,
            "layer_plans": [[0, [1]], [1, [4]]],
        }
    )
    assert covered == 4.0


def test_compute_fit_recovers_route_coefficient() -> None:
    rows = []
    for group in range(2):
        for token_count, uncovered in ((256, 4000), (512, 7000), (768, 9000)):
            total = 10.0 + group * 3.0 + token_count * 0.02 + uncovered * 0.015
            rows.append(
                {
                    "group_id": group,
                    "token_count": token_count,
                    "uncovered_route_entries": uncovered,
                    "total_ms": total,
                }
            )
    model = fit_compute_model(
        rows,
        rows,
        target="total_ms",
        token_scale=256.0,
        route_scale=8192.0,
    )
    assert abs(model["route_entry_gain_ms"] - 0.015) < 1e-9
    assert model["test"]["mae_ms"] < 1e-9


def test_transport_fit_recovers_components() -> None:
    rows = []
    for h2d, d2d in ((16, 0), (4, 12), (8, 8), (0, 0)):
        rows.append(
            {
                "h2d_experts": h2d,
                "d2d_experts": d2d,
                "host_prepare_ms": h2d * 4.5,
                "copy_event_ms": h2d * 0.5 + d2d * 0.08,
            }
        )
    model = fit_transport_model(rows, rows)
    assert abs(model["h2d_effective_ms_per_expert"] - 5.0) < 1e-9
    assert abs(model["d2d_effective_ms_per_expert"] - 0.08) < 1e-9


def test_action_log_alignment() -> None:
    text = "\n".join(
        [
            "[kt-stage] action_dispatched ticket=3 states=[1] group=2 tokens=256 layers=[8,10)",
            "[kt-time] layer=8 step=2 total=10.00ms submit=1.00 mask=1.00 gpu=1.00 sync=6.00 merge=1.00 cpu_wait=5.50ms num_tokens=256 cpu_entries=0 gpu_entries=0 cpu_unique=0 gpu_unique=0",
            "[kt-time] layer=9 step=2 total=20.00ms submit=2.00 mask=2.00 gpu=2.00 sync=12.00 merge=2.00 cpu_wait=11.00ms num_tokens=256 cpu_entries=0 gpu_entries=0 cpu_unique=0 gpu_unique=0",
            "[kt-stage] action_completed ticket=3 group=2 layers=[8,10) final=False",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "server.log"
        path.write_text(text + "\n", encoding="utf-8")
        actions, issues = parse_action_timing(path)
    assert not issues
    assert actions[3]["timing_complete"]
    assert actions[3]["total_ms"] == 30.0
    assert actions[3]["cpu_wait_ms"] == 16.5


def test_gate_rejects_missing_repeat_holdouts() -> None:
    rows = [
        {
            "primary_calibration_row": True,
        }
        for _ in range(120)
    ]
    valid_holdout = {
        "holdout": "fifo",
        "route_entry_gain_ms": 0.01,
        "test": {"normalized_mae": 0.01},
    }
    compute = {
        "policy_holdouts": {
            "cpu_wait_ms": [valid_holdout],
            "total_ms": [valid_holdout],
        },
        "repeat_holdouts": {
            "cpu_wait_ms": [],
            "total_ms": [],
        },
    }
    metadata = {
        "provenance": {"args": {"seq_len": 2048, "num_prompts": 8}},
        "repeats": [1, 2, 3],
        "policies": ["fifo", "min_delta", "cost_oracle"],
        "issues": [],
    }
    transport = {
        "all_rows": {
            "h2d_effective_ms_per_expert": 1.0,
            "d2d_effective_ms_per_expert": 0.1,
        }
    }
    gate = calibration_gate(
        rows,
        metadata,
        compute,
        transport,
        max_normalized_mae=0.25,
    )
    assert not gate["authorized"]
    assert "cpu_wait_ms has no repeat holdout validation" in gate["reasons"]
    assert "total_ms has no repeat holdout validation" in gate["reasons"]


if __name__ == "__main__":
    test_oracle_coverage_recomputes_frequency_hits()
    test_compute_fit_recovers_route_coefficient()
    test_transport_fit_recovers_components()
    test_action_log_alignment()
    test_gate_rejects_missing_repeat_holdouts()
    print("M20 cost calibration smoke: PASS")
