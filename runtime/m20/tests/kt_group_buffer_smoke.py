#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sglang"
    / "srt"
    / "layers"
    / "moe"
    / "kt_group_expert_buffer.py"
)
SPEC = importlib.util.spec_from_file_location("kt_group_expert_buffer_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

GroupBufferState = MODULE.GroupBufferState
GroupCopyKind = MODULE.GroupCopyKind
KTGroupBufferConfig = MODULE.KTGroupBufferConfig
KTGroupExpertBufferManager = MODULE.KTGroupExpertBufferManager


def load_wrapper_module_for_mask_test():
    """Load the local helper without importing the complete SGLang package."""

    source = (
        Path(__file__).resolve().parents[1]
        / "sglang"
        / "srt"
        / "layers"
        / "moe"
        / "kt_ep_wrapper.py"
    )
    text = source.read_text(encoding="utf-8")
    start = text.index("def _generate_group_mode_initial_masks(")
    end = text.index("\ndef _init_kt_gpu_experts_masks", start)
    namespace = {"torch": torch, "Optional": __import__("typing").Optional}
    exec(text[start:end], namespace)
    return (
        namespace["_generate_group_mode_initial_masks"],
        namespace["_fixed_slots_per_layer_from_server_args"],
    )


class FakeLayer(torch.nn.Module):
    def __init__(self, slots: int):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.empty(slots, 4, 3, dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(slots, 3, 2, dtype=torch.bfloat16), requires_grad=False
        )


class FakeWrapper:
    def __init__(self, slots: int, num_experts: int):
        self.num_gpu_experts = slots
        self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
        self.gpu_experts_mask[:slots] = True
        self.logical_to_gpu_index = torch.full(
            (num_experts,), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[:slots] = torch.arange(slots, dtype=torch.int32)
        self.gpu_index_to_logical = torch.arange(slots, dtype=torch.int32)
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.clone()
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.clone()
        self.wrapper = SimpleNamespace(gpu_experts_mask=self.gpu_experts_mask.clone())


class FakeProvider:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.step = -1

    def begin_step_if_layer0(self, layer_idx: int, num_tokens: int) -> None:
        assert layer_idx == 0
        assert num_tokens > 0
        self.step += 1

    def ranked_experts_for_active_step(self, *, target_layer: int, limit: int):
        start = (target_layer + self.step + 1) % self.num_experts
        return [(start + i) % self.num_experts for i in range(limit)]


class FakeLoader:
    def __init__(self):
        self.prefetch_calls = []

    def prefetch_experts(self, layer_idx: int, expert_ids):
        self.prefetch_calls.append((int(layer_idx), tuple(int(x) for x in expert_ids)))
        return len(expert_ids)

    def prepare_expert_for_group_slot(
        self,
        *,
        layer_idx: int,
        logical_expert_id: int,
        w13_dst: torch.Tensor,
        w2_dst: torch.Tensor,
    ):
        value = float(layer_idx * 20 + logical_expert_id)
        w13 = torch.full_like(w13_dst, value, device="cpu")
        w2 = torch.full_like(w2_dst, value, device="cpu")
        return {
            "source": "fake",
            "h2d_bytes": (
                w13_dst.numel() * w13_dst.element_size()
                + w2_dst.numel() * w2_dst.element_size()
            ),
            "prefetch_wait_ms": 0.0,
            "tensors": (w13[:2], w13[2:], w2),
            "keepalive": [w13, w2],
            "async_h2d_capable": False,
        }

    def enqueue_prepared_group_expert(
        self,
        *,
        prepared,
        w13_dst: torch.Tensor,
        w2_dst: torch.Tensor,
        stream,
    ):
        assert stream is None
        gate, up, down = prepared["tensors"]
        w13_dst[:2].copy_(gate)
        w13_dst[2:].copy_(up)
        w2_dst.copy_(down)
        return {
            **prepared,
            "keepalive": [gate, up, down],
        }


class PendingEvent:
    def __init__(self):
        self.ready = False

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.ready = True


class BlockingLoader(FakeLoader):
    def __init__(self, blocked_layer: int):
        super().__init__()
        self.blocked_layer = int(blocked_layer)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.prepare_calls = []

    def prepare_expert_for_group_slot(self, **kwargs):
        layer_idx = int(kwargs["layer_idx"])
        logical_expert_id = int(kwargs["logical_expert_id"])
        self.prepare_calls.append((layer_idx, logical_expert_id))
        if layer_idx == self.blocked_layer and not self.release.is_set():
            self.entered.set()
            if not self.release.wait(timeout=5.0):
                raise TimeoutError("pipeline smoke did not release host preparation")
        return super().prepare_expert_for_group_slot(**kwargs)


def build_manager(
    *,
    num_layers: int,
    group_size: int,
    slots: int,
    miss_policy: str = "block",
    load_mode: str = "async",
    materialization: str = "full",
    max_replacements: int | None = None,
):
    num_experts = 11
    manager = KTGroupExpertBufferManager(
        KTGroupBufferConfig(
            num_layers=num_layers,
            num_experts=num_experts,
            group_size=group_size,
            slots_per_layer=slots,
            buffer_count=2,
            load_mode=load_mode,
            miss_policy=miss_policy,
            prefetch_policy="oracle",
            oracle_required=True,
            require_bf16=True,
            materialization=materialization,
            max_replacements=max_replacements,
        ),
        strict_runtime=False,
    )
    layers = []
    wrappers = []
    for layer_idx in range(num_layers):
        layer = FakeLayer(slots)
        wrapper = FakeWrapper(slots, num_experts)
        manager.register_layer(layer_idx, layer, wrapper)
        layers.append(layer)
        wrappers.append(wrapper)
    return manager, layers, wrappers, FakeProvider(num_experts), FakeLoader()


def test_parameterized_layout_and_order() -> None:
    manager, layers, wrappers, provider, loader = build_manager(
        num_layers=5, group_size=2, slots=2
    )
    assert manager.physical_slots == 8
    assert tuple(manager.weight_buffers["w13_weight"].shape[:3]) == (2, 2, 2)
    assert tuple(manager.weight_buffers["w2_weight"].shape[:3]) == (2, 2, 2)

    manager.begin_step(num_tokens=17, loader=loader, provider=provider)
    for group_id in range(manager.config.num_groups):
        start, end = manager.config.group_range(group_id)
        assert manager.activate_group(group_id) == "gpu"
        for layer_idx in range(start, end):
            manager.assert_layer_ready_for_apply(layer_idx, layers[layer_idx])
            selected = manager.current_plans[layer_idx]
            assert tuple(wrappers[layer_idx].gpu_index_to_logical.tolist()) == selected
            for slot, expert_id in enumerate(selected):
                expected = float(layer_idx * 20 + expert_id)
                actual_w13 = float(layers[layer_idx].w13_weight[slot, 0, 0])
                actual_w2 = float(layers[layer_idx].w2_weight[slot, 0, 0])
                assert actual_w13 == expected, (
                    layer_idx,
                    slot,
                    expert_id,
                    actual_w13,
                    expected,
                )
                assert actual_w2 == expected, (
                    layer_idx,
                    slot,
                    expert_id,
                    actual_w2,
                    expected,
                )
        if group_id == 0:
            try:
                manager._enqueue_group_load(2, 0, asynchronous=True)
            except RuntimeError as exc:
                assert "ACTIVE buffer" in str(exc)
            else:
                raise AssertionError("ACTIVE buffer overwrite was not rejected")
        manager.finish_group(group_id)
    metrics = manager.end_step()
    assert metrics["groups_executed"] == 3
    assert metrics["commits"] == 3
    assert metrics["oracle_hits"] == 5
    assert metrics["oracle_misses"] == 0
    assert metrics["active_overwrite_rejections"] == 1
    assert metrics["host_prepare_ms"] >= 0
    assert metrics["prefetch_wait_ms"] == 0


def test_cpu_fallback_never_exposes_loading_weights() -> None:
    manager, layers, wrappers, provider, loader = build_manager(
        num_layers=2, group_size=1, slots=3, miss_policy="cpu_fallback"
    )
    manager.begin_step(num_tokens=9, loader=loader, provider=provider)
    assert manager.activate_group(0) == "gpu"
    manager.finish_group(0)

    record = manager.records[1]
    assert record.group_id == 1
    record.state = GroupBufferState.LOADING
    record.ready_event = PendingEvent()
    assert manager.activate_group(1) == "cpu_fallback"
    manager.assert_layer_ready_for_apply(1, layers[1])
    assert not bool(wrappers[1].gpu_experts_mask.any().item())
    assert not bool(wrappers[1].wrapper.gpu_experts_mask.any().item())
    manager.finish_group(1)
    metrics = manager.end_step()
    assert metrics["cpu_fallback_groups"] == 1
    assert metrics["ready_misses"] == 1


def test_host_prefetch_only_submits_future_cpu_work() -> None:
    manager, _layers, _wrappers, provider, loader = build_manager(
        num_layers=3, group_size=1, slots=2
    )
    manager.begin_step(num_tokens=9, loader=loader, provider=provider)
    assert manager.activate_group(0) == "gpu"
    manager.submit_host_prefetch_for_next_group(0)
    assert loader.prefetch_calls == [(1, manager.current_plans[1])]
    assert manager.records[1].state == GroupBufferState.EMPTY
    manager.finish_group(0)
    assert manager.records[1].group_id == 1
    manager.finish_group(1) if manager.activate_group(1) == "gpu" else None
    assert manager.activate_group(2) == "gpu"
    manager.finish_group(2)
    manager.end_step()


def test_h2d_prefetch_uses_only_inactive_buffer() -> None:
    manager, _layers, _wrappers, provider, loader = build_manager(
        num_layers=3, group_size=1, slots=2
    )
    manager.begin_step(num_tokens=9, loader=loader, provider=provider)
    assert manager.activate_group(0) == "gpu"
    manager.submit_host_prefetch_for_next_group(0)
    manager.enqueue_next_group_h2d_after_host_prefetch(0)
    assert manager.records[1].group_id == 1
    assert manager.records[1].state == GroupBufferState.READY
    assert manager.records[0].state == GroupBufferState.ACTIVE
    manager.finish_group(0)
    assert manager.activate_group(1) == "gpu"
    manager.finish_group(1)
    assert manager.activate_group(2) == "gpu"
    manager.finish_group(2)
    manager.end_step()


def test_cpu_fallback_defers_unsafe_next_prefetch() -> None:
    manager, _layers, _wrappers, provider, loader = build_manager(
        num_layers=4, group_size=1, slots=2, miss_policy="cpu_fallback"
    )
    manager.begin_step(num_tokens=9, loader=loader, provider=provider)
    assert manager.activate_group(0) == "gpu"
    manager.finish_group(0)

    # Group 1 is the scheduled next group in buffer 1. Keep it pending so the
    # fallback branch must not reuse that physical storage for group 2.
    record = manager.records[1]
    assert record.group_id == 1
    record.state = GroupBufferState.LOADING
    record.ready_event = PendingEvent()
    assert manager.activate_group(1) == "cpu_fallback"
    manager.finish_group(1)

    # Group 2 uses buffer 0 safely. Its completion would normally prefetch
    # group 3 into buffer 1, but buffer 1 is still loading group 1.
    assert manager.activate_group(2) == "gpu"
    manager.finish_group(2)
    assert manager.records[1].group_id == 1
    assert manager.records[1].state == GroupBufferState.LOADING
    assert manager.stats["prefetch_deferred_buffer_busy"] == 1

    # Group 3 must fall back rather than overwrite the pending buffer.
    assert manager.activate_group(3) == "cpu_fallback"
    manager.finish_group(3)
    metrics = manager.end_step()
    assert metrics["cpu_fallback_groups"] == 2
    assert metrics["prefetch_deferred_buffer_busy"] == 1


def test_capacity_formula_across_configs() -> None:
    for num_layers, group_size, slots in ((5, 1, 2), (5, 5, 3), (7, 4, 1)):
        manager, *_ = build_manager(
            num_layers=num_layers, group_size=group_size, slots=slots
        )
        assert manager.physical_slots == 2 * group_size * slots
        assert manager.config.num_groups == (
            num_layers + group_size - 1
        ) // group_size
        tail_start, tail_end = manager.config.group_range(manager.config.num_groups - 1)
        assert 0 < tail_end - tail_start <= group_size


def test_stage_actions_restore_batch_owned_plans() -> None:
    manager, layers, wrappers, provider, loader = build_manager(
        num_layers=4,
        group_size=2,
        slots=2,
        load_mode="sync",
    )
    batch_a_plans = manager.begin_action(
        group_id=0,
        num_tokens=17,
        loader=loader,
        provider=provider,
    )
    assert manager.activate_group(0) == "gpu"
    manager.finish_group(0)
    manager.end_action(0)

    batch_b_plans = manager.begin_action(
        group_id=0,
        num_tokens=13,
        loader=loader,
        provider=provider,
    )
    assert batch_b_plans != batch_a_plans
    assert manager.activate_group(0) == "gpu"
    manager.finish_group(0)
    manager.end_action(0)

    restored = manager.begin_action(
        group_id=0,
        num_tokens=17,
        loader=loader,
        provider=provider,
        plans=batch_a_plans,
    )
    assert restored == batch_a_plans
    assert manager.activate_group(0) == "gpu"
    for layer_idx in range(0, 2):
        manager.assert_layer_ready_for_apply(layer_idx, layers[layer_idx])
        assert tuple(wrappers[layer_idx].gpu_index_to_logical.tolist()) == (
            batch_a_plans[layer_idx]
        )
    manager.finish_group(0)
    metrics = manager.end_action(0)
    assert metrics["actions"] == 3
    assert metrics["groups_executed"] == 3


def run_explicit_action(manager, loader, group_id: int, plans, ticket_id: int):
    frozen = manager.begin_action(
        group_id=group_id,
        num_tokens=17,
        loader=loader,
        provider=None,
        plans=plans,
        ticket_id=ticket_id,
    )
    assert manager.activate_group(group_id) == "gpu"
    materialization = manager.action_materialization_snapshot()
    manager.finish_group(group_id)
    metrics = manager.end_action(group_id)
    assert metrics["action_materialization"] == materialization
    return frozen, materialization


def assert_action_weights(manager, layers, plans, group_id: int) -> None:
    start, end = manager.config.group_range(group_id)
    for layer_idx in range(start, end):
        selected = plans[layer_idx]
        for slot, expert_id in enumerate(selected):
            expected = float(layer_idx * 20 + expert_id)
            assert float(layers[layer_idx].w13_weight[slot, 0, 0]) == expected
            assert float(layers[layer_idx].w2_weight[slot, 0, 0]) == expected


def test_delta_materialization_four_paths_and_k_budget() -> None:
    manager, layers, wrappers, _provider, loader = build_manager(
        num_layers=2,
        group_size=2,
        slots=4,
        load_mode="sync",
        materialization="delta",
        max_replacements=1,
    )
    manager._integrity_check_enabled = True
    plan_a = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    plan_b = {0: (4, 1, 2, 3), 1: (4, 1, 2, 3)}
    plan_c = {0: (4, 1, 2, 5), 1: (4, 1, 2, 5)}

    frozen, cold = run_explicit_action(manager, loader, 0, plan_a, 0)
    assert frozen == plan_a
    assert cold["materialization"] == "delta"
    assert cold["source_buffer_id"] is None
    assert cold["target_buffer_id"] == 0
    assert cold["h2d_experts"] == 8
    assert cold["d2d_experts"] == 0
    assert_action_weights(manager, layers, plan_a, 0)

    frozen, zero = run_explicit_action(manager, loader, 0, plan_a, 1)
    assert frozen == plan_a
    assert zero["materialization"] == "zero"
    assert zero["target_buffer_id"] == 0
    assert zero["h2d_bytes"] == zero["d2d_bytes"] == 0
    assert zero["copy_ops"] == []

    frozen, first_delta = run_explicit_action(manager, loader, 0, plan_b, 2)
    assert frozen == plan_b
    assert first_delta["source_buffer_id"] == 0
    assert first_delta["target_buffer_id"] == 1
    assert first_delta["max_changed"] == 1
    assert first_delta["h2d_experts"] == 2
    assert first_delta["d2d_experts"] == 6
    assert first_delta["h2d_experts"] <= sum(
        first_delta["changed_by_layer"].values()
    )
    assert_action_weights(manager, layers, plan_b, 0)

    frozen, mixed = run_explicit_action(manager, loader, 0, plan_c, 3)
    assert frozen == plan_c
    assert mixed["source_buffer_id"] == 1
    assert mixed["target_buffer_id"] == 0
    assert mixed["max_changed"] == 1
    kinds = [GroupCopyKind(op["kind"]) for op in mixed["copy_ops"]]
    assert kinds.count(GroupCopyKind.RETAIN) == 4
    assert kinds.count(GroupCopyKind.D2D) == 2
    assert kinds.count(GroupCopyKind.H2D) == 2
    assert mixed["h2d_experts"] <= sum(mixed["changed_by_layer"].values())
    assert_action_weights(manager, layers, plan_c, 0)
    for layer_idx in range(2):
        assert tuple(wrappers[layer_idx].gpu_index_to_logical.tolist()) == plan_c[layer_idx]
    assert manager.stats["zero_loads"] == 1
    assert manager.stats["integrity_checks"] > 0


def test_generated_delta_plan_is_bounded_and_explicit_plan_fails_closed() -> None:
    for materialization in ("full", "delta"):
        manager, _layers, _wrappers, provider, loader = build_manager(
            num_layers=2,
            group_size=2,
            slots=4,
            load_mode="sync",
            materialization=materialization,
            max_replacements=1,
        )
        initial = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
        run_explicit_action(manager, loader, 0, initial, 0)
        generated = manager.begin_action(
            group_id=0,
            num_tokens=17,
            loader=loader,
            provider=provider,
            ticket_id=1,
        )
        for layer_idx in range(2):
            assert len(set(generated[layer_idx]) - set(initial[layer_idx])) <= 1
        manager.abort_step("bounded-plan-smoke")

        invalid = {0: (4, 5, 2, 3), 1: (4, 5, 2, 3)}
        try:
            manager.begin_action(
                group_id=0,
                num_tokens=17,
                loader=loader,
                provider=None,
                plans=invalid,
                ticket_id=2,
            )
        except RuntimeError as exc:
            assert "replacement budget" in str(exc)
            manager.abort_step("expected-invalid-explicit-plan")
        else:
            raise AssertionError("an explicit plan exceeding K was accepted")


def test_explicit_ticket_uses_frozen_logical_source_plan() -> None:
    manager, _layers, _wrappers, _provider, loader = build_manager(
        num_layers=4,
        group_size=2,
        slots=4,
        load_mode="sync",
        materialization="full",
        max_replacements=1,
    )
    plan_a = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    plan_b = {0: (0, 1, 4, 3), 1: (0, 1, 4, 3)}
    plan_c = {0: (0, 1, 5, 3), 1: (0, 1, 5, 3)}
    run_explicit_action(manager, loader, 0, plan_a, 0)
    frozen_b = manager.begin_action(
        group_id=0,
        num_tokens=17,
        loader=loader,
        provider=None,
        plans=plan_b,
        ticket_id=1,
        logical_source_plans=plan_a,
        logical_source_buffer_id=0,
    )
    assert frozen_b == plan_b
    assert manager.activate_group(0) == "gpu"
    manager.finish_group(0)
    manager.end_action(0)

    # The full path overwrote buffer 0 with plan B. A delta path would still
    # retain plan A in buffer 0 and plan B in buffer 1. The next ticket is
    # therefore bounded against the explicitly frozen logical source plan B,
    # independent of the full worker's latest physical source heuristic.
    frozen_c = manager.begin_action(
        group_id=0,
        num_tokens=17,
        loader=loader,
        provider=None,
        plans=plan_c,
        ticket_id=2,
        logical_source_plans=plan_b,
        logical_source_buffer_id=1,
    )
    assert frozen_c == plan_c
    manager.abort_step("logical-source-contract-smoke")


def _action_payload(
    manager,
    ticket_id: int,
    group_id: int,
    plans: dict,
    logical_source_plans: dict | None = None,
) -> dict:
    return {
        "ticket_id": int(ticket_id),
        "group_id": int(group_id),
        "layer_plans": [
            [layer_idx, list(experts)]
            for layer_idx, experts in sorted(plans.items())
        ],
        "plan_hash": manager._plan_hash(group_id, plans),
        "logical_source_plans": [
            [layer_idx, list(experts)]
            for layer_idx, experts in sorted(
                (logical_source_plans or {}).items()
            )
        ],
        "logical_source_buffer_id": None,
    }


def test_ticket_pipeline_end_is_nonblocking_and_next_action_adopts() -> None:
    manager, layers, _wrappers, _provider, _loader = build_manager(
        num_layers=2,
        group_size=1,
        slots=2,
        load_mode="async",
        materialization="full",
    )
    loader = BlockingLoader(blocked_layer=1)
    plan0 = {0: (0, 1)}
    plan1 = {1: (2, 3)}

    manager.begin_action(
        group_id=0,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=plan0,
        ticket_id=10,
    )
    assert manager.activate_group(0) == "gpu"
    assert manager.prefetch_action(_action_payload(manager, 11, 1, plan1))
    assert loader.entered.wait(timeout=2.0)
    manager.finish_group(0)
    started = time.perf_counter()
    manager.end_action(0)
    assert (time.perf_counter() - started) < 0.5
    assert manager._pending_action is not None

    loader.release.set()
    manager.begin_action(
        group_id=1,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=plan1,
        ticket_id=11,
    )
    assert manager._pending_action is None
    assert manager.activate_group(1) == "gpu"
    materialization = manager.action_materialization_snapshot()
    assert materialization["ticket_id"] == 11
    assert materialization["target_buffer_id"] == 1
    assert materialization["h2d_experts"] == 2
    assert len([call for call in loader.prepare_calls if call[0] == 1]) == 2
    assert_action_weights(manager, layers, plan1, 1)
    manager.finish_group(1)
    metrics = manager.end_action(1)
    assert metrics["pipeline_prefetch_submitted"] == 1
    assert metrics["pipeline_prefetch_adopted"] == 1
    assert metrics["pipeline_end_action_nonblocking"] == 2


def test_ticket_pipeline_mismatch_invalidates_and_falls_back() -> None:
    manager, _layers, _wrappers, _provider, loader = build_manager(
        num_layers=2,
        group_size=1,
        slots=2,
        load_mode="async",
        materialization="full",
    )
    plan0 = {0: (0, 1)}
    prefetched_plan = {1: (2, 3)}
    actual_plan = {1: (4, 5)}
    manager.begin_action(
        group_id=0,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=plan0,
        ticket_id=20,
    )
    assert manager.activate_group(0) == "gpu"
    manager.prefetch_action(_action_payload(manager, 21, 1, prefetched_plan))
    manager.finish_group(0)
    manager.end_action(0)

    manager.begin_action(
        group_id=1,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=actual_plan,
        ticket_id=22,
    )
    assert manager.stats["pipeline_prefetch_mismatches"] == 1
    assert manager.activate_group(1) == "gpu"
    materialization = manager.action_materialization_snapshot()
    assert materialization["ticket_id"] == 22
    manager.finish_group(1)
    manager.end_action(1)


def test_ticket_pipeline_abort_drains_background_materialization() -> None:
    manager, _layers, _wrappers, _provider, _loader = build_manager(
        num_layers=2,
        group_size=1,
        slots=2,
        load_mode="async",
        materialization="full",
    )
    loader = BlockingLoader(blocked_layer=1)
    plan0 = {0: (0, 1)}
    plan1 = {1: (2, 3)}
    manager.begin_action(
        group_id=0,
        num_tokens=8,
        loader=loader,
        provider=None,
        plans=plan0,
        ticket_id=30,
    )
    assert manager.activate_group(0) == "gpu"
    manager.prefetch_action(_action_payload(manager, 31, 1, plan1))
    assert loader.entered.wait(timeout=2.0)

    abort = threading.Thread(target=manager.abort_step, args=("pipeline-smoke",))
    abort.start()
    abort.join(timeout=0.1)
    assert abort.is_alive()
    loader.release.set()
    abort.join(timeout=2.0)
    assert not abort.is_alive()
    assert not manager.has_pending_action
    assert not manager.step_active
    assert all(record.state != GroupBufferState.LOADING for record in manager.records)


def test_sync_and_async_replay_freeze_identical_copy_ops() -> None:
    sequence = [
        (40, 0, {0: (0, 1)}, None),
        (41, 1, {1: (2, 3)}, None),
        (42, 0, {0: (0, 4)}, {0: (0, 1)}),
    ]

    def run(load_mode: str):
        manager, _layers, _wrappers, _provider, loader = build_manager(
            num_layers=2,
            group_size=1,
            slots=2,
            load_mode=load_mode,
            materialization="delta",
            max_replacements=2,
        )
        snapshots = []
        for index, (ticket, group_id, plans, source_plans) in enumerate(sequence):
            manager.begin_action(
                group_id=group_id,
                num_tokens=8,
                loader=loader,
                provider=None,
                plans=plans,
                ticket_id=ticket,
                logical_source_plans=source_plans,
            )
            assert manager.activate_group(group_id) == "gpu"
            snapshots.append(manager.action_materialization_snapshot())
            if index + 1 < len(sequence):
                next_ticket, next_group, next_plans, next_source = sequence[index + 1]
                manager.prefetch_action(
                    _action_payload(
                        manager,
                        next_ticket,
                        next_group,
                        next_plans,
                        next_source,
                    )
                )
            manager.finish_group(group_id)
            manager.end_action(group_id)
        return snapshots

    sync = run("sync")
    async_ = run("async")
    assert len(sync) == len(async_)
    for sync_action, async_action in zip(sync, async_):
        for field in (
            "ticket_id",
            "group_id",
            "plan_hash",
            "source_buffer_id",
            "target_buffer_id",
            "copy_ops",
            "h2d_experts",
            "d2d_experts",
            "retained_experts",
        ):
            assert sync_action[field] == async_action[field], field
    # A trace may deliberately have no abstract source-buffer binding even
    # when replay finds a physical source for delta materialization.  That
    # frozen ``None`` must not be rewritten by the sync path.
    assert sync[2]["logical_source_buffer_id"] is None
    assert async_[2]["logical_source_buffer_id"] is None


def test_sync_and_async_replay_survive_logical_source_eviction() -> None:
    plan_a = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    plan_b = {0: (4, 1, 2, 3), 1: (4, 1, 2, 3)}
    other = {2: (0, 1, 2, 3), 3: (0, 1, 2, 3)}
    plan_c = {0: (4, 1, 2, 5), 1: (4, 1, 2, 5)}
    plan_d = {0: (4, 1, 6, 5), 1: (4, 1, 6, 5)}
    sequence = [
        (50, 0, plan_a, None),
        (51, 0, plan_b, plan_a),
        (52, 0, plan_a, plan_b),
        (53, 1, other, None),
        (54, 0, plan_c, plan_b),
        (55, 0, plan_d, plan_c),
    ]

    def run(load_mode: str):
        manager, _layers, _wrappers, _provider, loader = build_manager(
            num_layers=4,
            group_size=2,
            slots=4,
            load_mode=load_mode,
            materialization="delta",
            max_replacements=1,
        )
        snapshots = []
        for index, (ticket, group_id, plans, source_plans) in enumerate(sequence):
            manager.begin_action(
                group_id=group_id,
                num_tokens=8,
                loader=loader,
                provider=None,
                plans=plans,
                ticket_id=ticket,
                logical_source_plans=source_plans,
            )
            assert manager.activate_group(group_id) == "gpu"
            snapshots.append(manager.action_materialization_snapshot())
            if index + 1 < len(sequence):
                next_ticket, next_group, next_plans, next_source = sequence[index + 1]
                manager.prefetch_action(
                    _action_payload(
                        manager,
                        next_ticket,
                        next_group,
                        next_plans,
                        next_source,
                    )
                )
            manager.finish_group(group_id)
            manager.end_action(group_id)
        return snapshots

    sync = run("sync")
    async_ = run("async")
    assert len(sync) == len(async_) == len(sequence)
    for sync_action, async_action in zip(sync, async_):
        for field in (
            "ticket_id",
            "group_id",
            "plan_hash",
            "source_buffer_id",
            "target_buffer_id",
            "physical_source_matches_logical",
            "copy_ops",
            "h2d_experts",
            "d2d_experts",
            "retained_experts",
        ):
            assert sync_action[field] == async_action[field], field

    evicted = sync[4]
    assert evicted["ticket_id"] == 54
    assert not evicted["physical_source_matches_logical"]
    assert evicted["h2d_experts"] == 4
    assert sum(evicted["changed_by_layer"].values()) == 2
    assert sync[1]["physical_source_matches_logical"]
    assert sync[5]["physical_source_matches_logical"]


def test_group_initial_mask_uses_slots_not_artifact_topk() -> None:
    generate, _ = load_wrapper_module_for_mask_test()
    activation = torch.arange(5 * 11, dtype=torch.float32).reshape(5, 11)
    masks = generate(
        strategy="frequency",
        activation_freq=activation,
        num_layers=5,
        num_experts=11,
        slots_per_layer=2,
        first_k_dense_replace=0,
        moe_layer_freq=1,
    )
    assert tuple(masks.shape) == (5, 11)
    assert masks.sum(dim=1).tolist() == [2, 2, 2, 2, 2]
    assert masks[:, -2:].all()


def test_explicit_static_count_is_fixed_per_layer() -> None:
    _, resolve_fixed_slots = load_wrapper_module_for_mask_test()
    static_count = SimpleNamespace(
        kt_slots_per_layer=4,
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=2,
    )
    ratio = SimpleNamespace(
        kt_slots_per_layer=4,
        kt_gpu_experts_ratio=0.1,
        kt_num_gpu_experts=2,
    )
    group = SimpleNamespace(
        kt_slots_per_layer=3,
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=2,
    )
    assert resolve_fixed_slots(static_count, group_mode=False) == 2
    assert resolve_fixed_slots(ratio, group_mode=False) is None
    assert resolve_fixed_slots(group, group_mode=True) == 3


def main() -> int:
    test_parameterized_layout_and_order()
    test_cpu_fallback_never_exposes_loading_weights()
    test_host_prefetch_only_submits_future_cpu_work()
    test_h2d_prefetch_uses_only_inactive_buffer()
    test_cpu_fallback_defers_unsafe_next_prefetch()
    test_capacity_formula_across_configs()
    test_stage_actions_restore_batch_owned_plans()
    test_delta_materialization_four_paths_and_k_budget()
    test_generated_delta_plan_is_bounded_and_explicit_plan_fails_closed()
    test_explicit_ticket_uses_frozen_logical_source_plan()
    test_ticket_pipeline_end_is_nonblocking_and_next_action_adopts()
    test_ticket_pipeline_mismatch_invalidates_and_falls_back()
    test_ticket_pipeline_abort_drains_background_materialization()
    test_sync_and_async_replay_freeze_identical_copy_ops()
    test_sync_and_async_replay_survive_logical_source_eviction()
    test_group_initial_mask_uses_slots_not_artifact_topk()
    test_explicit_static_count_is_fixed_per_layer()
    print("kt group buffer smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
