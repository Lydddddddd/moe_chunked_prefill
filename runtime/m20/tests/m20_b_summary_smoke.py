#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


PREDICTOR_DIR = Path(__file__).resolve().parents[1] / "inter_layer_predictor"
sys.path.insert(0, str(PREDICTOR_DIR))

from summarize_m20_b import (  # noqa: E402
    aggregate,
    aggregate_pairs,
    formal_gate,
    paired_delta_comparisons,
    predictor_gate,
    write_report,
)


def make_row(policy: str, repeat: int, throughput: float) -> dict:
    return {
        "policy_label": policy,
        "stage_policy": "fifo" if policy == "frequency" else policy,
        "materialization": "delta",
        "repeat": repeat,
        "status": "ok",
        "run_name": f"r{repeat}_{policy}_delta",
        "client": {
            "prefill_tokens_per_s": throughput,
            "latency_p50_s": 10.0 / throughput,
            "latency_p95_s": 11.0 / throughput,
        },
        "memory": {
            "baseline": {"compute_process_count": 0},
            "peak_memory_used_mb": 8000,
        },
        "group_profile": {
            "event_counts": {"action_end": 10},
            "last_action_metrics": {
                "h2d_bytes": 1000,
                "d2d_bytes": 100,
                "zero_loads": 2,
                "host_prepare_ms": 5.0,
                "h2d_ms": 1.0,
            },
        },
    }


def test_matched_frequency_pairs_and_predictor_gate() -> None:
    rows = []
    for repeat in (1, 2, 3):
        for policy, throughput in (
            ("frequency", 100.0),
            ("fifo", 101.0),
            ("min_delta", 102.0),
            ("cost_oracle", 104.0),
        ):
            delta = make_row(policy, repeat, throughput)
            rows.append(delta)
            rows.append({**delta, "materialization": "full"})
    correctness = {"passed": True}
    provenance = {
        "args": {
            "seq_len": 2048,
            "num_prompts": 8,
            "repeats": 3,
            "resource_mode": "exclusive",
            "rotate_policy_order": True,
            "prompt_offset": 128,
            "warmup_num_prompts": 8,
            "warmup_prompt_offset": 136,
        },
        "resource_snapshot": {"has_smt_siblings": False},
    }
    calibration = {"gate": {"authorized": True, "reasons": []}}
    gate = formal_gate(rows, provenance, correctness, calibration)
    assert gate["formal_run_eligible"]
    assert gate["cost_oracle_conclusion_authorized"]

    pair_rows = paired_delta_comparisons(rows)
    pairs = aggregate_pairs(pair_rows)
    cost = next(row for row in pairs if row["policy"] == "cost_oracle")
    assert abs(cost["throughput_speedup_mean_pct"] - 4.0) < 1e-9
    assert predictor_gate(gate, pairs)["proceed_to_predictor"]

    aggregates = aggregate(rows)
    frequency = next(
        row
        for row in aggregates
        if row["policy"] == "frequency" and row["materialization"] == "delta"
    )
    assert frequency["repeat_count"] == 3
    assert frequency["zero_loads_mean"] == 2.0

    result = {
        "generated_at": "test",
        "formal_gate": gate,
        "predictor_gate": predictor_gate(gate, pairs),
        "aggregates": aggregates,
        "paired_aggregates": pairs,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.md"
        write_report(report_path, result)
        report = report_path.read_text(encoding="utf-8")
        assert "| tok/s stddev |" in report
        assert f"{frequency['throughput_stdev']:.3f}" in report


def test_gate_rejects_unapproved_calibration() -> None:
    rows = [
        make_row(policy, repeat, 100.0)
        for repeat in (1, 2, 3)
        for policy in ("frequency", "fifo", "min_delta", "cost_oracle")
    ]
    rows += [{**row, "materialization": "full"} for row in list(rows)]
    gate = formal_gate(
        rows,
        {
            "args": {
                "seq_len": 2048,
                "num_prompts": 8,
                "repeats": 3,
                "resource_mode": "exclusive",
                "rotate_policy_order": True,
                "prompt_offset": 128,
                "warmup_num_prompts": 8,
                "warmup_prompt_offset": 136,
            },
            "resource_snapshot": {"has_smt_siblings": False},
        },
        {"passed": True},
        {"gate": {"authorized": False, "reasons": ["bad fit"]}},
    )
    assert gate["formal_run_eligible"]
    assert not gate["cost_oracle_conclusion_authorized"]
    assert gate["calibration_reasons"] == ["bad fit"]


if __name__ == "__main__":
    test_matched_frequency_pairs_and_predictor_gate()
    test_gate_rejects_unapproved_calibration()
    print("M20-B summary smoke: PASS")
