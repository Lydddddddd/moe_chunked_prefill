#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path


PREDICTOR_DIR = Path(__file__).resolve().parents[1] / "inter_layer_predictor"
sys.path.insert(0, str(PREDICTOR_DIR))

from run_m20_b import (  # noqa: E402
    audit_replacement_budget,
    acquire_exclusive_lock,
    build_pipeline_acceptance,
    build_specs,
    compare_pipeline_replays,
    load_resumable_row,
    policy_order_for_repeat,
    replay_modes_for_repeat,
    spec_run_name,
)


def make_row(*materializations: dict) -> dict:
    return {
        "group_profile": {
            "action_metrics": [
                {"action_materialization": materialization}
                for materialization in materializations
            ]
        }
    }


def test_logical_and_physical_audit_scopes() -> None:
    audit = audit_replacement_budget(
        make_row(
            {
                "logical_source_plan_present": True,
                "logical_source_buffer_id": None,
                "source_buffer_id": None,
                "materialization": "delta",
                "changed_by_layer": {"0": 1, "1": 1},
                "h2d_experts": 8,
            },
            {
                "logical_source_plan_present": True,
                "logical_source_buffer_id": 0,
                "source_buffer_id": 0,
                "materialization": "delta",
                "changed_by_layer": {"0": 1, "1": 1},
                "h2d_experts": 2,
            },
            {
                "logical_source_plan_present": True,
                "logical_source_buffer_id": 0,
                "source_buffer_id": 0,
                "materialization": "zero",
                "changed_by_layer": {"0": 0, "1": 0},
                "h2d_experts": 0,
            },
        ),
        max_replacements=1,
    )
    assert audit == {
        "passed": True,
        "checked_logical_source_actions": 3,
        "checked_physical_source_actions": 1,
        "violations": [],
    }


def test_cold_action_still_enforces_logical_k() -> None:
    audit = audit_replacement_budget(
        make_row(
            {
                "ticket_id": 7,
                "logical_source_plan_present": True,
                "logical_source_buffer_id": None,
                "source_buffer_id": None,
                "materialization": "delta",
                "changed_by_layer": {"0": 2},
                "h2d_experts": 4,
            }
        ),
        max_replacements=1,
    )
    assert not audit["passed"]
    assert audit["checked_logical_source_actions"] == 1
    assert audit["checked_physical_source_actions"] == 0
    assert audit["violations"][0]["ticket_id"] == 7
    assert "exceeds K=1" in audit["violations"][0]["reasons"][0]


def test_delta_source_enforces_h2d_bound() -> None:
    audit = audit_replacement_budget(
        make_row(
            {
                "ticket_id": 8,
                "logical_source_plan_present": True,
                "logical_source_buffer_id": 1,
                "source_buffer_id": 1,
                "materialization": "delta",
                "changed_by_layer": {"4": 1, "5": 1},
                "h2d_experts": 3,
            }
        ),
        max_replacements=1,
    )
    assert not audit["passed"]
    assert audit["checked_logical_source_actions"] == 1
    assert audit["checked_physical_source_actions"] == 1
    assert "exceeds changed=2" in audit["violations"][0]["reasons"][0]


def test_evicted_logical_source_allows_physical_cache_miss_h2d() -> None:
    audit = audit_replacement_budget(
        make_row(
            {
                "ticket_id": 9,
                "logical_source_plan_present": True,
                "logical_source_buffer_id": 1,
                "physical_source_matches_logical": False,
                "source_buffer_id": 0,
                "materialization": "delta",
                "changed_by_layer": {"0": 1, "1": 1},
                "h2d_experts": 4,
            }
        ),
        max_replacements=1,
    )
    assert audit == {
        "passed": True,
        "checked_logical_source_actions": 1,
        "checked_physical_source_actions": 0,
        "violations": [],
    }


def test_frequency_baseline_is_matched_static_k0_pair() -> None:
    class Args:
        group_size = 4
        slots = 4
        max_replacements = 1
        placement = "oracle"
        policies = ["fifo", "min_delta", "cost_oracle"]
        include_frequency_baseline = True

    specs = build_specs(Args())
    assert set(specs) == {"fifo", "min_delta", "cost_oracle", "frequency"}
    frequency = specs["frequency"]
    assert [item["materialization"] for item in frequency] == ["full", "delta"]
    assert all(item["placement"] == "static" for item in frequency)
    assert all(item["stage_policy"] == "fifo" for item in frequency)
    assert all(item["policy_label"] == "frequency" for item in frequency)
    assert all(item["max_replacements"] == 0 for item in frequency)
    assert all(item["physical_slots"] == 32 for item in frequency)


def test_policy_pairs_rotate_deterministically() -> None:
    policies = ["fifo", "min_delta", "cost_oracle", "frequency"]
    assert policy_order_for_repeat(policies, 1, rotate=True) == policies
    assert policy_order_for_repeat(policies, 2, rotate=True) == [
        "cost_oracle",
        "frequency",
        "fifo",
        "min_delta",
    ]
    assert policy_order_for_repeat(policies, 3, rotate=True) == [
        "frequency",
        "fifo",
        "min_delta",
        "cost_oracle",
    ]
    assert policy_order_for_repeat(policies, 2, rotate=False) == policies


def test_async_replay_spec_is_explicit_and_has_distinct_run_name() -> None:
    class Args:
        group_size = 4
        slots = 4
        max_replacements = 1
        placement = "oracle"
        policies = ["fifo"]
        include_frequency_baseline = False
        replay_load_mode = "async"

    planner, replay = build_specs(Args())["fifo"]
    assert "load_mode" not in planner
    assert replay["load_mode"] == "async"
    assert spec_run_name(replay, 1).endswith("_k1_async")


def test_both_replay_specs_share_planner_and_rotate_order() -> None:
    class Args:
        group_size = 4
        slots = 8
        max_replacements = 1
        placement = "oracle"
        policies = ["min_delta"]
        include_frequency_baseline = False
        replay_load_mode = "both"

    planner, sync, async_spec = build_specs(Args())["min_delta"]
    assert "load_mode" not in planner
    assert [sync["load_mode"], async_spec["load_mode"]] == ["sync", "async"]
    assert spec_run_name(sync, 1).endswith("_k1")
    assert spec_run_name(async_spec, 1).endswith("_k1_async")
    assert replay_modes_for_repeat("both", 1) == ["sync", "async"]
    assert replay_modes_for_repeat("both", 2) == ["async", "sync"]
    assert replay_modes_for_repeat("both", 3) == ["sync", "async"]


def test_pipeline_acceptance_compares_exact_materialization_and_gate() -> None:
    materialization = {
        "ticket_id": 0,
        "group_id": 0,
        "plan_hash": "plan",
        "target_buffer_id": 0,
        "source_buffer_id": None,
        "copy_ops": [{"kind": "h2d", "expert_id": 7}],
        "h2d_experts": 1,
        "d2d_experts": 0,
        "h2d_bytes": 16,
        "d2d_bytes": 0,
        "retained_experts": 0,
        "retained_bytes": 0,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def make_pipeline_row(name: str, mode: str, throughput: float) -> dict:
            result_dir = root / name
            result_dir.mkdir()
            (result_dir / "default_request_metrics.jsonl").write_text(
                json.dumps({"prompt_index": 128, "generated_text": "ok"}) + "\n",
                encoding="utf-8",
            )
            return {
                "status": "ok",
                "repeat": 1,
                "run_name": name,
                "result_dir": str(result_dir),
                "load_mode": mode,
                "policy_label": "min_delta",
                "client": {
                    "prefill_tokens_per_s": throughput,
                    "latency_p50_s": 10.0 / throughput,
                },
                "memory": {
                    "baseline": {"device_memory_used_mb": 4},
                    "peak_memory_used_mb": 100,
                    "peak_device_memory_used_mb": 104,
                },
                "group_profile": {
                    "event_counts": {"action_end": 1},
                    "action_metrics": [
                        {"action_materialization": dict(materialization)}
                    ],
                    "last_action_metrics": {
                        "h2d_experts": 1,
                        "d2d_experts": 0,
                        "h2d_bytes": 16,
                        "d2d_bytes": 0,
                        "retained_experts": 0,
                        "retained_bytes": 0,
                        "pipeline_prefetch_submitted": 0,
                        "pipeline_prefetch_adopted": 0,
                        "pipeline_end_action_nonblocking": 1,
                        "pipeline_prefetch_failures": 0,
                        "pipeline_prefetch_mismatches": 0,
                        # Residual event tails are performance coverage, not
                        # replay correctness failures, provided each miss is
                        # accounted by the formal block policy.
                        "ready_misses": 2,
                        "block_count": 2,
                        "active_overwrite_rejections": 0,
                    },
                },
            }

        sync = make_pipeline_row("sync", "sync", 100.0)
        async_row = make_pipeline_row("async", "async", 105.0)
        pair = compare_pipeline_replays(sync, async_row)
        assert pair["outputs_match"]
        assert pair["materializations_match"]
        assert pair["copy_ops_match"]
        assert pair["pipeline_counters_passed"]
        assert pair["pipeline_counters"]["ready_misses"] == 2
        assert pair["pipeline_counters"]["blocks"] == 2
        assert abs(pair["throughput_speedup_pct"] - 5.0) < 1e-9

        pairs = []
        rows = []
        for repeat in (1, 2, 3):
            repeat_pair = dict(pair, repeat=repeat)
            pairs.append(repeat_pair)
            rows.extend([dict(sync, repeat=repeat), dict(async_row, repeat=repeat)])

        class FormalArgs:
            repeats = 3
            resource_mode = "exclusive"
            seq_len = 2048
            num_prompts = 8
            concurrency = 8
            warmup_num_prompts = 8
            pin_cpu_tensors = True

        acceptance = build_pipeline_acceptance(
            pairs, rows, FormalArgs(), expected_pairs=3
        )
        assert acceptance["correctness_passed"]
        assert acceptance["formal_eligible"]
        assert acceptance["performance_passed"]
        assert acceptance["passed"]


def test_resume_only_reuses_completed_rows() -> None:
    class Args:
        group_size = 4
        slots = 4
        max_replacements = 1
        placement = "oracle"
        policies = ["fifo"]
        include_frequency_baseline = False

    spec = build_specs(Args())["fifo"][0]
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        run_name = (
            f"r1_{spec['mode']}_g{spec['group_size']}_s{spec['slots_per_layer']}"
            f"_k{spec['max_replacements']}"
        )
        status_path = output_dir / run_name / "runner_status.json"
        status_path.parent.mkdir(parents=True)

        assert load_resumable_row(output_dir, spec, 1) is None
        status_path.write_text(
            json.dumps({"status": "failed", "error": "interrupted"}),
            encoding="utf-8",
        )
        assert load_resumable_row(output_dir, spec, 1) is None
        completed = {
            "status": "ok",
            "run_name": run_name,
            "result_dir": str(status_path.parent),
        }
        status_path.write_text(json.dumps(completed), encoding="utf-8")
        assert load_resumable_row(output_dir, spec, 1) == completed


def test_exclusive_lock_rejects_smt_and_competing_runner() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / "formal.lock"
        try:
            acquire_exclusive_lock(
                lock_path,
                {
                    "has_smt_siblings": True,
                    "physical_core_count": 64,
                },
                64,
            )
        except RuntimeError as exc:
            assert "one logical CPU" in str(exc)
        else:
            raise AssertionError("SMT-overlapping affinity was accepted")

        snapshot = {
            "has_smt_siblings": False,
            "physical_core_count": 64,
            "logical_cpus": list(range(64)),
        }
        first = acquire_exclusive_lock(
            lock_path, snapshot, 64, check_competing_runners=False
        )
        try:
            try:
                acquire_exclusive_lock(
                    lock_path,
                    snapshot,
                    64,
                    check_competing_runners=False,
                )
            except RuntimeError as exc:
                assert "another exclusive" in str(exc)
            else:
                raise AssertionError("competing exclusive runner was accepted")
        finally:
            first.close()


if __name__ == "__main__":
    test_logical_and_physical_audit_scopes()
    test_cold_action_still_enforces_logical_k()
    test_delta_source_enforces_h2d_bound()
    test_evicted_logical_source_allows_physical_cache_miss_h2d()
    test_frequency_baseline_is_matched_static_k0_pair()
    test_policy_pairs_rotate_deterministically()
    test_async_replay_spec_is_explicit_and_has_distinct_run_name()
    test_both_replay_specs_share_planner_and_rotate_order()
    test_pipeline_acceptance_compares_exact_materialization_and_gate()
    test_resume_only_reuses_completed_rows()
    test_exclusive_lock_rejects_smt_and_competing_runner()
    print("M20-B runner smoke: PASS")
