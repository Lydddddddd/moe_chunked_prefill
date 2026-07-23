#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import http.client
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "inter_layer_predictor"
    / "run_m20_a0_a1.py"
)
SPEC = importlib.util.spec_from_file_location("run_m20_a0_a1_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
def write_metrics(root: Path, name: str, token: str) -> Path:
    result_dir = root / name
    result_dir.mkdir()
    (result_dir / "default_request_metrics.jsonl").write_text(
        json.dumps({"prompt_index": 128, "generated_text": token}) + "\n",
        encoding="utf-8",
    )
    return result_dir


def test_correctness_reference_is_matched_by_slot_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for slots, token in ((2, "slot-2"), (4, "slot-4")):
            baseline_name = f"r1_b0_oracle_foreground_g1_s{slots}"
            rows.append(
                {
                    "mode": "b0_oracle_foreground",
                    "slots_per_layer": slots,
                    "repeat": 1,
                    "run_name": baseline_name,
                    "status": "ok",
                    "result_dir": str(write_metrics(root, baseline_name, token)),
                }
            )
            candidate_name = f"r1_o1_async_block_g2_s{slots}"
            rows.append(
                {
                    "mode": "o1_async_block",
                    "slots_per_layer": slots,
                    "repeat": 1,
                    "run_name": candidate_name,
                    "status": "ok",
                    "result_dir": str(write_metrics(root, candidate_name, token)),
                }
            )

        result = MODULE.compare_outputs(rows)
        assert result["all_match"] is True
        assert result["complete"] is True
        assert len(result["comparisons"]) == 4
        for comparison in result["comparisons"]:
            candidate_slots = comparison["candidate"].rsplit("_s", 1)[1]
            baseline_slots = comparison["baseline"].rsplit("_s", 1)[1]
            assert candidate_slots == baseline_slots


def test_failed_candidate_makes_correctness_incomplete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline_name = "r1_b0_oracle_foreground_g5_s2"
        rows = [
            {
                "mode": "b0_oracle_foreground",
                "slots_per_layer": 2,
                "repeat": 1,
                "run_name": baseline_name,
                "status": "ok",
                "result_dir": str(write_metrics(root, baseline_name, "token")),
            },
            {
                "mode": "b2_group_sync",
                "slots_per_layer": 2,
                "repeat": 1,
                "run_name": "r1_b2_group_sync_g5_s2",
                "status": "failed",
                "error": "interrupted",
                "result_dir": str(root / "failed"),
            },
        ]
        result = MODULE.compare_outputs(rows)
        assert result["all_match"] is False
        assert result["complete"] is False
        assert result["skipped"][0]["candidate"] == "r1_b2_group_sync_g5_s2"


def test_compute_process_memory_parser() -> None:
    assert MODULE.parse_compute_process_memory("") == (0, 0)
    assert MODULE.parse_compute_process_memory("123, 7452\n456, 101.5\n") == (
        7553,
        2,
    )


def test_model_metadata_hashes_are_scoped_to_small_contract_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text('{"model_type":"qwen3_moe"}\n')
        (root / "model-00001-of-00016.safetensors").write_bytes(b"weight")
        hashes = MODULE.model_metadata_hashes(root)
        assert set(hashes) == {"model/config.json"}


def test_stage_fifo_uses_oracle_reference_and_bounded_capacity() -> None:
    specs = MODULE.build_specs(
        Namespace(
            modes="b0_oracle_foreground,b0_stage_fifo",
            slots=[4],
            group_sizes=[4],
        )
    )
    MODULE.validate_correctness_matrix(specs)
    stage = next(row for row in specs if row["mode"] == "b0_stage_fifo")
    assert stage["physical_slots"] == 2 * 4 * 4


def test_stage_fifo_static_requires_matched_static_reference() -> None:
    specs = MODULE.build_specs(
        Namespace(
            modes="b0_static,b0_stage_fifo_static",
            slots=[4],
            group_sizes=[4],
        )
    )
    MODULE.validate_correctness_matrix(specs)
    stage = next(row for row in specs if row["mode"] == "b0_stage_fifo_static")
    assert stage["physical_slots"] == 2 * 4 * 4

    missing_reference = MODULE.build_specs(
        Namespace(
            modes="b0_stage_fifo_static",
            slots=[4],
            group_sizes=[4],
        )
    )
    try:
        MODULE.validate_correctness_matrix(missing_reference)
    except ValueError as exc:
        assert "b0_static" in str(exc)
    else:
        raise AssertionError("static stage mode accepted without b0_static reference")


def test_server_probe_retries_transient_non_http_response() -> None:
    class Opener:
        def open(self, *_args, **_kwargs):
            raise http.client.BadStatusLine("startup socket")

    class Process:
        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 1

    with mock.patch.object(
        MODULE.urllib.request, "build_opener", return_value=Opener()
    ), mock.patch.object(MODULE.time, "sleep", return_value=None):
        assert MODULE.wait_server(32920, 1.0, Process()) is False


if __name__ == "__main__":
    test_correctness_reference_is_matched_by_slot_count()
    test_failed_candidate_makes_correctness_incomplete()
    test_compute_process_memory_parser()
    test_model_metadata_hashes_are_scoped_to_small_contract_files()
    test_stage_fifo_uses_oracle_reference_and_bounded_capacity()
    test_stage_fifo_static_requires_matched_static_reference()
    test_server_probe_retries_transient_non_http_response()
    print("M20 runner logic smoke: PASS")
