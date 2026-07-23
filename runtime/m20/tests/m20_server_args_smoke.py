#!/usr/bin/env python3
from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.server_args import ServerArgs


@contextmanager
def expect_value_error(message: str):
    try:
        yield
    except ValueError as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError(f"expected ValueError containing: {message}")


def make_args() -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    args.kt_group_expert_buffer = True
    args.kt_group_load_mode = "sync"
    args.kt_stage_ready_scheduler = True
    args.kt_stage_policy = "fifo"
    args.kt_stage_cohort_size = 1
    args.kt_stage_candidate_window = 1
    args.kt_stage_max_consecutive = 2
    args.kt_stage_max_wait_ms = 0.0
    args.kt_stage_max_inflight_chunks = 8
    args.kt_action_trace_path = None
    args.kt_action_replay_path = None
    args.prefill_max_requests = None
    args.enable_mixed_chunk = False
    return args


def test_valid_b0a_contract() -> None:
    make_args()._handle_kt_stage_ready_scheduler()


def test_b1b_policies_and_cost_contract() -> None:
    for policy in ("fifo", "min_delta", "cost_oracle"):
        args = make_args()
        args.kt_stage_policy = policy
        args._handle_kt_stage_ready_scheduler()

    args = make_args()
    args.kt_stage_route_entry_gain_ms = -0.1
    with expect_value_error("must be non-negative"):
        args._handle_kt_stage_ready_scheduler()

    args = make_args()
    args.kt_stage_confidence_threshold = 1.1
    with expect_value_error("must be in [0, 1]"):
        args._handle_kt_stage_ready_scheduler()


def test_group_delta_materialization_bounds() -> None:
    args = make_args()
    args.kt_weight_path = "dummy.gguf"
    args.kt_method = "LLAMAFILE"
    args.tp_size = 1
    args.pp_size = 1
    args.kt_slots_per_layer = 4
    args.kt_group_materialization = "delta"
    args.kt_group_max_replacements = 2
    args.kt_gpu_experts_ratio = None
    args.kt_num_gpu_experts = 4
    args.kt_enable_dynamic_expert_update = False
    args.kt_lora_path = None
    args.kt_expert_lora_path = None
    args.enable_pdmux = False
    args.kt_group_prefetch_policy = "static"
    args.kt_group_oracle_required = False
    args.disable_cuda_graph = True
    args._handle_kt_group_expert_buffer()
    assert args.kt_group_max_replacements == 2

    args.kt_group_max_replacements = 5
    with expect_value_error("must be in"):
        args._handle_kt_group_expert_buffer()


def test_group_full_materialization_accepts_bounded_logical_plan() -> None:
    args = make_args()
    args.kt_weight_path = "dummy.gguf"
    args.kt_method = "LLAMAFILE"
    args.tp_size = 1
    args.pp_size = 1
    args.kt_slots_per_layer = 4
    args.kt_group_materialization = "full"
    args.kt_group_max_replacements = 1
    args.kt_gpu_experts_ratio = None
    args.kt_num_gpu_experts = 4
    args.kt_enable_dynamic_expert_update = False
    args.kt_lora_path = None
    args.kt_expert_lora_path = None
    args.enable_pdmux = False
    args.kt_group_prefetch_policy = "static"
    args.kt_group_oracle_required = False
    args.disable_cuda_graph = True
    args._handle_kt_group_expert_buffer()
    assert args.kt_group_max_replacements == 1


def test_disabled_trace_is_rejected() -> None:
    args = make_args()
    args.kt_stage_ready_scheduler = False
    args.kt_action_trace_path = "actions.jsonl"
    with expect_value_error("require --kt-stage-ready-scheduler"):
        args._handle_kt_stage_ready_scheduler()


def test_stage_scheduler_requires_group_buffers() -> None:
    args = make_args()
    args.kt_group_expert_buffer = False
    with expect_value_error("requires --kt-group-expert-buffer"):
        args._handle_kt_stage_ready_scheduler()


def test_b0a_bounds() -> None:
    invalid_values = (
        ("kt_stage_cohort_size", 0, "must be positive"),
        ("kt_stage_candidate_window", 0, "must be at least"),
        ("kt_stage_max_consecutive", 0, "must be positive"),
        ("kt_stage_max_wait_ms", -0.1, "cannot be negative"),
        ("kt_stage_max_inflight_chunks", 0, "must be positive"),
    )
    for field, value, message in invalid_values:
        args = make_args()
        setattr(args, field, value)
        with expect_value_error(message):
            args._handle_kt_stage_ready_scheduler()


def test_async_stage_pipeline_requires_frozen_replay() -> None:
    args = make_args()
    args.kt_group_load_mode = "async"
    args.kt_group_miss_policy = "block"
    with expect_value_error("requires --kt-action-replay-path"):
        args._handle_kt_stage_ready_scheduler()

    args.kt_action_replay_path = "actions.jsonl"
    args._handle_kt_stage_ready_scheduler()


def test_async_stage_pipeline_requires_block_policy() -> None:
    args = make_args()
    args.kt_group_load_mode = "async"
    args.kt_group_miss_policy = "cpu_fallback"
    args.kt_action_replay_path = "actions.jsonl"
    with expect_value_error("requires --kt-group-miss-policy block"):
        args._handle_kt_stage_ready_scheduler()


def test_trace_and_replay_are_exclusive() -> None:
    args = make_args()
    args.kt_action_trace_path = "write.jsonl"
    args.kt_action_replay_path = "read.jsonl"
    with expect_value_error("mutually exclusive"):
        args._handle_kt_stage_ready_scheduler()


def test_stage_ownership_forces_single_request_admission() -> None:
    args = make_args()
    args.kt_stage_cohort_size = 4
    args.kt_stage_candidate_window = 8
    args._handle_kt_stage_ready_scheduler()
    assert args.prefill_max_requests == 1


if __name__ == "__main__":
    test_valid_b0a_contract()
    test_b1b_policies_and_cost_contract()
    test_group_delta_materialization_bounds()
    test_group_full_materialization_accepts_bounded_logical_plan()
    test_disabled_trace_is_rejected()
    test_stage_scheduler_requires_group_buffers()
    test_b0a_bounds()
    test_async_stage_pipeline_requires_frozen_replay()
    test_async_stage_pipeline_requires_block_policy()
    test_trace_and_replay_are_exclusive()
    test_stage_ownership_forces_single_request_admission()
    print("M20 server args smoke: PASS")
