#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sglang.srt.managers.kt_stage_scheduler import (
    StageScheduler,
    StageSchedulerConfig,
    StageState,
    compute_plan_hash,
    normalize_layer_plans,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.kt_stage_batch import StageExecutionContext


class FakeSamplingBatchInfo:
    @classmethod
    def from_schedule_batch(cls, batch, vocab_size):
        return cls()


def fake_pack_stage_batches(payloads, group_id, group_size):
    batch = payloads[0].batch
    spans = ((0, int(batch.extend_num_tokens)),)
    return batch, StageExecutionContext(group_id, tuple(payloads), spans)


def fake_demux_stage_batch(execution, context):
    for payload in context.payloads:
        payload.batch.split_forward_batch = execution.split_forward_batch
        payload.batch.split_index = int(execution.split_forward_batch.split_index)


scheduler_module.SamplingBatchInfo = FakeSamplingBatchInfo
scheduler_module.pack_stage_batches = fake_pack_stage_batches
scheduler_module.demux_stage_batch = fake_demux_stage_batch


def make_runtime() -> Scheduler:
    runtime = object.__new__(Scheduler)
    runtime._kt_stage_ready_enabled = True
    runtime._kt_fifo_split_group_size = 2
    runtime._kt_fifo_split_batch = None
    runtime._kt_stage_scheduler = StageScheduler(
        StageSchedulerConfig(
            num_groups=2,
            cohort_size=1,
            candidate_window=1,
            max_consecutive=2,
            max_inflight_states=4,
            policy="fifo",
        )
    )
    runtime._kt_stage_replay = None
    runtime._kt_stage_trace_writer = None
    runtime._kt_stage_next_state_id = 0
    runtime._kt_stage_execution_contexts = {}
    runtime._kt_stage_pending_continuations = []
    runtime.chunked_req = None
    runtime._kt_multi_chunked_reqs = []
    runtime._chunked_req_scheduled_last_iter = False
    runtime.model_config = SimpleNamespace(num_hidden_layers=4, vocab_size=100)
    runtime.server_args = SimpleNamespace(kt_stage_max_wait_ms=0.0)
    return runtime


def make_batch(prompt_index: int, token_start: int = 0) -> ScheduleBatch:
    req = SimpleNamespace(
        rid=f"runtime-{prompt_index}",
        is_chunked=0,
        kt_metadata={"prompt_global_index": prompt_index},
    )
    return ScheduleBatch(
        reqs=[req],
        forward_mode=ForwardMode.EXTEND,
        prefix_lens=[token_start],
        extend_lens=[8],
        extend_num_tokens=8,
    )


def complete_action(runtime: Scheduler, batch: ScheduleBatch, next_index: int) -> bool:
    ticket = batch._kt_stage_ticket
    start = ticket.group_id * runtime._kt_fifo_split_group_size
    plans = normalize_layer_plans(
        {
            layer_idx: (0, 1, 2, 3)
            for layer_idx in range(
                start,
                min(
                    start + runtime._kt_fifo_split_group_size,
                    runtime.model_config.num_hidden_layers,
                ),
            )
        }
    )
    batch.split_forward_batch = SimpleNamespace(
        split_index=next_index,
        kt_metadata_list=[dict(req.kt_metadata or {}) for req in batch.reqs],
        _kt_stage_materialized_action={
            "ticket_id": ticket.ticket_id,
            "group_id": ticket.group_id,
            "layer_plans": [
                [layer_idx, list(experts)] for layer_idx, experts in plans
            ],
            "plan_hash": compute_plan_hash(ticket.group_id, plans),
        },
    )
    return runtime._advance_kt_stage_batch(batch)


def test_interleaved_batch_ownership() -> None:
    runtime = make_runtime()
    batch_a = make_batch(10)
    batch_b = make_batch(11)

    runtime._admit_kt_stage_batch(batch_a)
    selected = runtime._select_kt_stage_batch()
    assert selected is batch_a
    assert selected._kt_stage_ticket.group_id == 0
    assert complete_action(runtime, batch_a, 2) is False

    runtime._admit_kt_stage_batch(batch_b)
    selected = runtime._select_kt_stage_batch()
    assert selected is batch_b
    assert selected._kt_stage_ticket.group_id == 0
    assert complete_action(runtime, batch_b, 2) is False

    selected = runtime._select_kt_stage_batch()
    assert selected is batch_a
    assert selected._kt_stage_ticket.group_id == 1
    assert complete_action(runtime, batch_a, 4) is True
    runtime._finalize_kt_stage_batch(batch_a)

    queues = runtime._kt_stage_scheduler.queues
    assert queues.states[batch_a._kt_stage_state_id].status == StageState.DONE
    assert queues.states[batch_b._kt_stage_state_id].status == StageState.READY
    assert queues.states[batch_b._kt_stage_state_id].group_id == 1
    queues.assert_invariants()


def test_batch_identity_is_metadata_stable() -> None:
    first = make_batch(42, token_start=256)
    second = make_batch(42, token_start=256)
    first.reqs[0].rid = "random-runtime-a"
    second.reqs[0].rid = "random-runtime-b"
    assert Scheduler._kt_stage_batch_identity(first) == (
        Scheduler._kt_stage_batch_identity(second)
    )


def test_deep_queue_continuations_keep_their_request_slots() -> None:
    runtime = make_runtime()
    runtime.running_batch = SimpleNamespace(batch_is_full=True)
    stashed = []
    runtime.stash_chunked_request = stashed.append
    continuations = []
    for index in range(16):
        req = type("FakeReq", (), {})()
        req.rid = f"continuation-{index}"
        req.req_pool_idx = index
        continuations.append(req)
    runtime._kt_stage_pending_continuations = list(continuations)

    excluded = set()
    for expected in continuations:
        activated = runtime._activate_next_kt_stage_continuation(excluded)
        assert activated is expected
        assert activated.req_pool_idx == continuations.index(expected)
        assert runtime.chunked_req is expected
        assert not runtime.running_batch.batch_is_full
        runtime.chunked_req = None  # Simulate ownership transfer at stage admission.

    assert stashed == continuations
    assert excluded == set(continuations)
    assert runtime._kt_stage_pending_continuations == []


def test_pending_continuation_triggers_admission_without_waiting_requests() -> None:
    runtime = make_runtime()
    runtime.last_batch = None
    runtime.waiting_queue = []
    runtime._kt_stage_pending_continuations = [SimpleNamespace(rid="pending")]
    sentinel = object()
    runtime._get_next_batch_to_run_legacy = lambda: sentinel
    runtime._select_kt_stage_batch = lambda: None
    assert runtime._get_next_kt_stage_batch_to_run() is sentinel


def test_replay_dispatch_carries_nonadvancing_next_action_hint() -> None:
    runtime = make_runtime()
    batch = make_batch(7)
    runtime._admit_kt_stage_batch(batch)
    next_payload = {
        "ticket_id": 99,
        "group_id": 1,
        "layer_plans": [[2, [0, 1, 2, 3]], [3, [0, 1, 2, 3]]],
        "plan_hash": "frozen-next-plan",
        "logical_source_plans": [],
        "logical_source_buffer_id": None,
    }

    class FakeReplay:
        def __init__(self):
            self.peek_calls = 0

        def reserve_next(self, _queues):
            return runtime._kt_stage_scheduler.choose_next()

        def peek_next(self):
            self.peek_calls += 1
            return SimpleNamespace(execution_payload=lambda: dict(next_payload))

    replay = FakeReplay()
    runtime._kt_stage_replay = replay
    selected = runtime._select_kt_stage_batch()
    assert selected is batch
    assert replay.peek_calls == 1
    for req in selected.reqs:
        assert req.kt_metadata["_kt_stage_next_action"] == next_payload
    if selected.split_forward_batch is not None:
        for metadata in selected.split_forward_batch.kt_metadata_list:
            assert metadata["_kt_stage_next_action"] == next_payload


def test_worker_submits_lookahead_between_activation_and_compute() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "sglang"
        / "srt"
        / "model_executor"
        / "model_runner.py"
    )
    source = source_path.read_text(encoding="utf-8")
    start = source.index("elif forward_batch.forward_mode.is_split_prefill():")
    end = source.index(
        "elif forward_batch.forward_mode.is_extend", start
    )
    block = source[start:end]
    activate = block.index("activate_kt_group(group_id)")
    prefetch = block.index("prefetch_kt_group_action(next_action)")
    compute = block.index("ret = self.forward_split_prefill", prefetch)
    finish = block.index("finish_kt_group(group_id)", compute)
    end_action = block.index("end_kt_group_action(group_id)", finish)
    assert activate < prefetch < compute < finish < end_action


if __name__ == "__main__":
    test_interleaved_batch_ownership()
    test_batch_identity_is_metadata_stable()
    test_deep_queue_continuations_keep_their_request_slots()
    test_pending_continuation_triggers_admission_without_waiting_requests()
    test_replay_dispatch_carries_nonadvancing_next_action_hint()
    test_worker_submits_lookahead_between_activation_and_compute()
    print("M20 stage runtime smoke: PASS")
