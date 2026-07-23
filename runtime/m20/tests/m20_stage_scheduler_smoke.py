#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sglang"
    / "srt"
    / "managers"
    / "kt_stage_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("kt_stage_scheduler_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ActionTrace = MODULE.ActionTrace
ActionReplayCursor = MODULE.ActionReplayCursor
ActionTraceWriter = MODULE.ActionTraceWriter
ChunkStageState = MODULE.ChunkStageState
CopyKind = MODULE.CopyKind
CopyOp = MODULE.CopyOp
NextActionTicket = MODULE.NextActionTicket
StageScheduler = MODULE.StageScheduler
StageSchedulerConfig = MODULE.StageSchedulerConfig
StageState = MODULE.StageState


def make_state(
    state_id: int,
    *,
    request_id: str | None = None,
    group_id: int = 0,
    enqueue_seq: int | None = None,
    ready_since: float = 0.0,
    deadline_at: float = 100.0,
    demand_by_layer=None,
    confidence: float = 1.0,
) -> ChunkStageState:
    return ChunkStageState(
        state_id=state_id,
        request_id=request_id or f"r{state_id}",
        chunk_index=0,
        token_start=state_id * 256,
        token_end=(state_id + 1) * 256,
        group_id=group_id,
        num_groups=3,
        enqueue_seq=state_id if enqueue_seq is None else enqueue_seq,
        ready_since=ready_since,
        deadline_at=deadline_at,
        payload={"state": state_id},
        demand_by_layer=demand_by_layer,
        confidence_by_layer=(
            None
            if demand_by_layer is None
            else {layer_idx: confidence for layer_idx in demand_by_layer}
        ),
    )


def build_scheduler(**overrides) -> StageScheduler:
    values = {
        "num_groups": 3,
        "cohort_size": 1,
        "candidate_window": 1,
        "max_consecutive": 2,
        "max_inflight_states": 8,
        "policy": "fifo",
    }
    values.update(overrides)
    return StageScheduler(
        StageSchedulerConfig(**values),
        plan_builder=lambda group_id, _states: {
            group_id * 4 + offset: (0, 1, 2, 3) for offset in range(4)
        },
        provider_version="smoke-v1",
    )


def run_ticket(scheduler: StageScheduler, ticket, now: float) -> bool:
    scheduler.mark_running(ticket)
    return scheduler.complete_group(ticket, now=now)


def build_working_set_scheduler(policy: str, **overrides) -> StageScheduler:
    frequency = tuple(
        (layer_idx, (0, 1, 2, 3)) for layer_idx in range(12)
    )
    values = {
        "num_groups": 3,
        "cohort_size": 2,
        "candidate_window": 3,
        "max_consecutive": 2,
        "max_inflight_states": 8,
        "policy": policy,
        "group_size": 4,
        "num_layers": 12,
        "slots_per_layer": 4,
        "max_replacements": 1,
        "frequency_plans": frequency,
        "expert_nbytes": 9 * 1024 * 1024,
        "h2d_expert_ms": 5.4,
        "d2d_expert_ms": 0.08,
    }
    values.update(overrides)
    return StageScheduler(StageSchedulerConfig(**values), provider_version="oracle-v1")


def repeated_demand(values: dict[int, float], start_layer: int = 0) -> dict:
    return {
        layer_idx: dict(values)
        for layer_idx in range(start_layer, start_layer + 4)
    }


def test_fifo_progress_and_no_global_barrier() -> None:
    scheduler = build_scheduler()
    scheduler.queues.admit(make_state(0, enqueue_seq=0))
    scheduler.queues.admit(make_state(1, enqueue_seq=1))

    first = scheduler.choose_next(now=1.0)
    assert first is not None and first.state_ids == (0,) and first.group_id == 0
    assert run_ticket(scheduler, first, 2.0) is False

    # State 0 is already ready for group 1, while state 1 has not left group 0.
    assert [state.state_id for state in scheduler.queues.ready_states(1)] == [0]
    assert [state.state_id for state in scheduler.queues.ready_states(0)] == [1]
    second = scheduler.choose_next(now=3.0)
    assert second is not None and second.state_ids == (1,) and second.group_id == 0
    run_ticket(scheduler, second, 4.0)
    scheduler.queues.assert_invariants()


def test_q_cap_and_deadline_override() -> None:
    scheduler = build_scheduler(max_consecutive=1)
    scheduler.queues.admit(make_state(0, group_id=0, enqueue_seq=0))
    scheduler.queues.admit(make_state(1, group_id=0, enqueue_seq=1))
    scheduler.queues.admit(make_state(2, group_id=1, enqueue_seq=2))

    first = scheduler.choose_next(now=1.0)
    assert first is not None and first.group_id == 0
    run_ticket(scheduler, first, 2.0)

    # Q=1 forces a different non-empty stage even though group 0 has an older head.
    second = scheduler.choose_next(now=3.0)
    assert second is not None and second.group_id == 1
    scheduler.unreserve(second, now=3.5)

    # A hard deadline overrides Q and any reuse preference.
    scheduler.queues.states[1].deadline_at = 3.6
    expired = scheduler.choose_next(now=4.0)
    assert expired is not None and expired.state_ids == (1,)
    assert expired.fallback == "deadline_fifo"
    scheduler.unreserve(expired, now=4.1)
    scheduler.queues.assert_invariants()


def test_partial_cohort_and_single_request_guard() -> None:
    scheduler = build_scheduler(cohort_size=4, candidate_window=8)
    scheduler.queues.admit(make_state(0))
    scheduler.queues.admit(make_state(1))
    ticket = scheduler.choose_next(now=1.0)
    assert ticket is not None and ticket.state_ids == (0, 1)
    assert ticket.token_count == 512
    try:
        scheduler.queues.admit(make_state(2, request_id="r0"))
    except RuntimeError as exc:
        assert "one in-flight chunk" in str(exc)
    else:
        raise AssertionError("same-request wavefront was not rejected")
    scheduler.unreserve(ticket, now=2.0)
    scheduler.queues.assert_invariants()


def test_reservation_versions_and_terminal_cleanup() -> None:
    scheduler = build_scheduler()
    scheduler.queues.admit(make_state(0))
    ticket = scheduler.choose_next(now=1.0)
    assert ticket is not None
    state = scheduler.queues.states[0]
    assert state.status == StageState.RESERVED
    assert state.state_version == ticket.state_versions[0]
    scheduler.mark_running(ticket)
    assert state.status == StageState.RUNNING
    scheduler.complete_group(ticket, now=2.0)

    for expected_group in (1, 2):
        next_ticket = scheduler.choose_next(now=3.0 + expected_group)
        assert next_ticket is not None and next_ticket.group_id == expected_group
        scheduler.mark_running(next_ticket)
        final = scheduler.complete_group(next_ticket, now=4.0 + expected_group)
    assert final is True and state.status == StageState.FINALIZING
    scheduler.queues.finalize([0])
    assert state.status == StageState.DONE
    assert "r0" not in scheduler.queues.active_request_ids
    scheduler.queues.assert_invariants()


def test_ticket_and_hash_chained_trace_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "action_trace.jsonl"
        with ActionTraceWriter(
            path, metadata={"runtime_hash": "abc", "policy": "fifo"}
        ) as writer:
            scheduler = build_scheduler()
            scheduler.trace_writer = writer
            scheduler.queues.admit(make_state(0))
            ticket = scheduler.choose_next(now=1.0)
            assert ticket is not None
            scheduler.mark_running(ticket)
            scheduler.complete_group(ticket, now=2.0)
            expected_hash = writer.trace_hash

        trace = ActionTrace.load(path)
        assert trace.metadata["runtime_hash"] == "abc"
        assert trace.trace_hash == expected_hash
        tickets = list(trace.iter_tickets())
        assert tickets == [ticket]
        assert NextActionTicket.from_dict(ticket.to_dict()) == ticket

        replay_scheduler = build_scheduler()
        replay = ActionReplayCursor(trace)
        assert replay.peek_next() == ticket
        assert replay.index == 0
        assert replay.reserve_next(replay_scheduler.queues) is None
        assert replay.peek_next() == ticket
        assert replay.index == 0
        replay_scheduler.queues.admit(make_state(0))
        replay_ticket = replay.reserve_next(replay_scheduler.queues)
        assert replay_ticket == ticket
        assert replay.peek_next() is None
        replay.assert_exhausted()
        replay_scheduler.queues.mark_running(replay_ticket)
        replay_scheduler.queues.complete_group(replay_ticket, now=2.0)
        replay_scheduler.queues.assert_invariants()

        rows = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(rows[1])
        tampered["ticket"]["policy"] = "tampered"
        rows[1] = json.dumps(tampered, sort_keys=True)
        bad_path = Path(tmp) / "tampered.jsonl"
        bad_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        try:
            ActionTrace.load(bad_path)
        except ValueError as exc:
            assert "hash" in str(exc)
        else:
            raise AssertionError("tampered action trace was accepted")


def test_materialized_trace_drives_replay_placement() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "materialized_trace.jsonl"
        with ActionTraceWriter(path) as writer:
            scheduler = StageScheduler(
                StageSchedulerConfig(num_groups=3), trace_writer=writer
            )
            state = make_state(0)
            scheduler.queues.admit(state)
            planned = scheduler.choose_next(now=1.0)
            assert planned is not None and not planned.layer_plans
            scheduler.mark_running(planned)
            plans = {layer_idx: (4, 5, 6, 7) for layer_idx in range(4)}
            materialized = scheduler.materialize_ticket(planned, plans)
            assert materialized.layer_plans
            scheduler.complete_group(materialized, now=2.0)

        trace = ActionTrace.load(path)
        assert not next(trace.iter_tickets("planned")).layer_plans
        recorded = next(trace.iter_tickets("materialized"))
        assert recorded == materialized
        replay = ActionReplayCursor(trace)
        assert replay.ticket_event == "materialized"
        assert replay.tickets == (materialized,)

        replay_scheduler = StageScheduler(StageSchedulerConfig(num_groups=3))
        replay_scheduler.queues.admit(make_state(0))
        replay_ticket = replay.reserve_next(replay_scheduler.queues)
        assert replay_ticket == materialized
        assert replay_ticket.plan_dict == plans


def test_copy_op_contract() -> None:
    op = CopyOp(
        kind=CopyKind.D2D,
        layer_idx=4,
        expert_id=7,
        src_buffer_id=0,
        src_slot=2,
        dst_buffer_id=1,
        dst_slot=3,
        nbytes=9 * 1024 * 1024,
    )
    assert CopyOp.from_dict(op.to_dict()) == op
    try:
        CopyOp(
            kind=CopyKind.D2D,
            layer_idx=0,
            expert_id=0,
            dst_buffer_id=1,
            dst_slot=0,
            nbytes=1,
        )
    except ValueError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("source-less D2D operation was accepted")


def test_materialized_ticket_carries_physical_delta_plan() -> None:
    scheduler = build_scheduler()
    scheduler.queues.admit(make_state(0))
    ticket = scheduler.choose_next(now=1.0)
    assert ticket is not None
    scheduler.mark_running(ticket)
    op = CopyOp(
        kind=CopyKind.H2D,
        layer_idx=0,
        expert_id=7,
        dst_buffer_id=1,
        dst_slot=0,
        nbytes=9 * 1024 * 1024,
    )
    materialized = scheduler.materialize_ticket(
        ticket,
        ticket.layer_plans,
        target_buffer_id=1,
        expected_buffer_versions=(3, 4),
        copy_ops=(op,),
    )
    assert materialized.target_buffer_id == 1
    assert materialized.expected_buffer_versions == (3, 4)
    assert materialized.copy_ops == (op,)
    assert NextActionTicket.from_dict(materialized.to_dict()) == materialized
    scheduler.complete_group(materialized, now=2.0)


def test_min_delta_partner_selection_keeps_stage_anchor() -> None:
    scheduler = build_working_set_scheduler("min_delta")
    scheduler.queues.admit(
        make_state(0, demand_by_layer=repeated_demand({0: 10, 1: 10, 2: 10, 3: 10}))
    )
    scheduler.queues.admit(
        make_state(1, demand_by_layer=repeated_demand({4: 100}))
    )
    scheduler.queues.admit(
        make_state(2, demand_by_layer=repeated_demand({0: 100}))
    )
    ticket = scheduler.choose_next(now=1.0)
    assert ticket is not None
    assert ticket.state_ids == (0, 2)
    assert all(experts == (0, 1, 2, 3) for _, experts in ticket.layer_plans)
    scheduler.unreserve(ticket, now=2.0)


def test_cost_oracle_nonpositive_gain_keeps_current_plan() -> None:
    scheduler = build_working_set_scheduler(
        "cost_oracle", cohort_size=1, route_entry_gain_ms=0.0
    )
    scheduler.queues.admit(
        make_state(0, demand_by_layer=repeated_demand({4: 100}))
    )
    ticket = scheduler.choose_next(now=1.0)
    assert ticket is not None
    assert ticket.fallback == "nonpositive_fifo_current"
    assert ticket.score.net_gain_ms < 0
    assert all(experts == (0, 1, 2, 3) for _, experts in ticket.layer_plans)
    scheduler.unreserve(ticket, now=2.0)


def test_cost_oracle_positive_gain_respects_k() -> None:
    scheduler = build_working_set_scheduler(
        "cost_oracle", cohort_size=1, route_entry_gain_ms=1.0
    )
    scheduler.queues.admit(
        make_state(0, demand_by_layer=repeated_demand({4: 100}))
    )
    ticket = scheduler.choose_next(now=1.0)
    assert ticket is not None
    assert ticket.fallback == ""
    assert ticket.score.net_gain_ms > 0
    assert ticket.score.covered_route_entries == 400
    assert ticket.score.compute_gain_ms == 400
    assert NextActionTicket.from_dict(ticket.to_dict()) == ticket
    for _layer_idx, experts in ticket.layer_plans:
        assert len(set(experts) - {0, 1, 2, 3}) == 1
    scheduler.unreserve(ticket, now=2.0)


def test_ticket_freezes_logical_delta_source_across_stage_overwrite() -> None:
    scheduler = build_working_set_scheduler("min_delta", cohort_size=1)
    scheduler.queues.admit(
        make_state(0, demand_by_layer=repeated_demand({4: 100}))
    )
    first = scheduler.choose_next(now=1.0)
    assert first is not None and first.group_id == 0
    assert dict(first.logical_source_plans) == {
        layer_idx: (0, 1, 2, 3) for layer_idx in range(4)
    }
    assert all(
        len(set(experts) - set(dict(first.logical_source_plans)[layer_idx])) <= 1
        for layer_idx, experts in first.layer_plans
    )
    run_ticket(scheduler, first, 2.0)

    stage_one = scheduler.choose_next(now=3.0)
    assert stage_one is not None and stage_one.group_id == 1
    run_ticket(scheduler, stage_one, 4.0)

    scheduler.queues.admit(
        make_state(1, demand_by_layer=repeated_demand({5: 100}))
    )
    next_group_zero = scheduler.choose_next(now=5.0)
    assert next_group_zero is not None and next_group_zero.group_id == 0
    source = dict(next_group_zero.logical_source_plans)
    for layer_idx, experts in next_group_zero.layer_plans:
        assert len(set(experts) - set(source[layer_idx])) <= 1
    scheduler.unreserve(next_group_zero, now=6.0)


def test_low_confidence_and_zero_load_reuse_fail_closed() -> None:
    scheduler = build_working_set_scheduler("fifo")
    scheduler.queues.admit(
        make_state(
            0,
            demand_by_layer=repeated_demand({4: 100}),
            confidence=0.5,
        )
    )
    first = scheduler.choose_next(now=1.0)
    assert first is not None
    assert first.fallback == "fifo"
    assert first.confidence == 0.5
    assert all(experts == (0, 1, 2, 3) for _, experts in first.layer_plans)
    run_ticket(scheduler, first, 2.0)

    scheduler.queues.admit(
        make_state(1, demand_by_layer=repeated_demand({0: 100}))
    )
    second = scheduler.choose_next(now=3.0)
    assert second is not None and second.group_id == 0
    assert second.copy_ops == ()
    scheduler.unreserve(second, now=4.0)


if __name__ == "__main__":
    test_fifo_progress_and_no_global_barrier()
    test_q_cap_and_deadline_override()
    test_partial_cohort_and_single_request_guard()
    test_reservation_versions_and_terminal_cleanup()
    test_ticket_and_hash_chained_trace_roundtrip()
    test_materialized_trace_drives_replay_placement()
    test_copy_op_contract()
    test_materialized_ticket_carries_physical_delta_plan()
    test_min_delta_partner_selection_keeps_stage_anchor()
    test_cost_oracle_nonpositive_gain_keeps_current_plan()
    test_cost_oracle_positive_gain_respects_k()
    test_ticket_freezes_logical_delta_source_across_stage_overwrite()
    test_low_confidence_and_zero_load_reuse_fail_closed()
    print("M20 stage scheduler smoke: PASS")
