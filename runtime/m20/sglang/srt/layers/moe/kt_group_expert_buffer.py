# SPDX-License-Identifier: Apache-2.0
"""Physical group-scoped expert buffers for the KT hybrid MoE runtime.

The manager owns the only GPU storage used by group-buffer mode. Decoder
layers keep lightweight Parameter views so the existing FusedMoE kernels and
checkpoint loader remain compatible, but all views point into the bounded
``buffer_count * group_size * slots_per_layer`` allocation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)


class GroupBufferState(str, Enum):
    EMPTY = "EMPTY"
    LOADING = "LOADING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    REUSABLE = "REUSABLE"


@dataclass(frozen=True)
class GroupActionSpec:
    """Immutable placement contract used by foreground and lookahead actions."""

    group_id: int
    plans: Tuple[Tuple[int, Tuple[int, ...]], ...]
    ticket_id: Optional[int] = None
    logical_source_plans: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()
    logical_source_buffer_id: Optional[int] = None
    plan_hash: Optional[str] = None

    @property
    def plan_dict(self) -> Dict[int, Tuple[int, ...]]:
        return {int(layer): tuple(experts) for layer, experts in self.plans}

    @property
    def logical_source_plan_dict(self) -> Optional[Dict[int, Tuple[int, ...]]]:
        if not self.logical_source_plans:
            return None
        return {
            int(layer): tuple(experts)
            for layer, experts in self.logical_source_plans
        }


@dataclass
class PendingGroupAction:
    spec: GroupActionSpec
    target_buffer_id: Optional[int]
    future: Future
    submitted_at: float
    zero_load: bool = False
    source_buffer_id: Optional[int] = None
    loader: Optional[Any] = None
    deferred_sync: bool = False


class GroupCopyKind(str, Enum):
    RETAIN = "retain"
    D2D = "d2d"
    H2D = "h2d"


@dataclass(frozen=True)
class GroupCopyOp:
    kind: GroupCopyKind
    layer_idx: int
    expert_id: int
    dst_buffer_id: int
    dst_slot: int
    nbytes: int
    src_buffer_id: Optional[int] = None
    src_slot: Optional[int] = None

    def __post_init__(self) -> None:
        if min(
            self.layer_idx,
            self.expert_id,
            self.dst_buffer_id,
            self.dst_slot,
            self.nbytes,
        ) < 0:
            raise ValueError("group copy operation fields must be non-negative")
        if self.kind == GroupCopyKind.D2D and (
            self.src_buffer_id is None or self.src_slot is None
        ):
            raise ValueError("D2D group copy operation requires a source slot")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "layer_idx": self.layer_idx,
            "expert_id": self.expert_id,
            "dst_buffer_id": self.dst_buffer_id,
            "dst_slot": self.dst_slot,
            "nbytes": self.nbytes,
            "src_buffer_id": self.src_buffer_id,
            "src_slot": self.src_slot,
        }


@dataclass(frozen=True)
class KTGroupBufferConfig:
    num_layers: int
    num_experts: int
    group_size: int
    slots_per_layer: int
    buffer_count: int = 2
    load_mode: str = "sync"
    miss_policy: str = "block"
    prefetch_policy: str = "oracle"
    oracle_required: bool = True
    require_bf16: bool = True
    materialization: str = "full"
    max_replacements: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.group_size <= self.num_layers:
            raise ValueError(
                f"group_size must be in [1, {self.num_layers}], got {self.group_size}"
            )
        if not 1 <= self.slots_per_layer <= self.num_experts:
            raise ValueError(
                "slots_per_layer must be in "
                f"[1, {self.num_experts}], got {self.slots_per_layer}"
            )
        if self.buffer_count != 2:
            raise ValueError(
                "M20-A implements double buffering only; buffer_count must be 2"
            )
        if self.load_mode not in {"sync", "async"}:
            raise ValueError("load_mode must be 'sync' or 'async'")
        if self.miss_policy not in {"block", "cpu_fallback"}:
            raise ValueError("miss_policy must be 'block' or 'cpu_fallback'")
        if self.prefetch_policy not in {"oracle", "static"}:
            raise ValueError("prefetch_policy must be 'oracle' or 'static'")
        if self.materialization not in {"full", "delta"}:
            raise ValueError("materialization must be 'full' or 'delta'")
        max_replacements = (
            self.slots_per_layer
            if self.max_replacements is None
            else int(self.max_replacements)
        )
        if not 0 <= max_replacements <= self.slots_per_layer:
            raise ValueError(
                "max_replacements must be in "
                f"[0, {self.slots_per_layer}], got {max_replacements}"
            )
        object.__setattr__(self, "max_replacements", max_replacements)

    @property
    def physical_slots(self) -> int:
        return self.buffer_count * self.group_size * self.slots_per_layer

    @property
    def num_groups(self) -> int:
        return (self.num_layers + self.group_size - 1) // self.group_size

    def group_range(self, group_id: int) -> Tuple[int, int]:
        if not 0 <= group_id < self.num_groups:
            raise IndexError(f"group_id out of range: {group_id}")
        start = group_id * self.group_size
        return start, min(start + self.group_size, self.num_layers)


@dataclass
class GroupBufferRecord:
    buffer_id: int
    state: GroupBufferState = GroupBufferState.EMPTY
    group_id: Optional[int] = None
    version: int = 0
    plan: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    plan_hash: Optional[str] = None
    ticket_id: Optional[int] = None
    content_version: int = 0
    slot_experts: Dict[int, Tuple[Optional[int], ...]] = field(default_factory=dict)
    valid_slots: Dict[int, Tuple[bool, ...]] = field(default_factory=dict)
    ready_event: Optional[Any] = None
    timing_start_event: Optional[Any] = None
    timing_end_event: Optional[Any] = None
    load_started_at: Optional[float] = None
    host_prepare_completed_at: Optional[float] = None
    load_enqueued_at: Optional[float] = None
    load_completed_at: Optional[float] = None
    activated_at: Optional[float] = None
    host_prepare_ms: float = 0.0
    prefetch_wait_ms: float = 0.0
    h2d_bytes: int = 0
    d2d_bytes: int = 0
    retained_bytes: int = 0
    h2d_experts: int = 0
    d2d_experts: int = 0
    retained_experts: int = 0
    materialization_kind: str = "full"
    copy_ops: List[Dict[str, Any]] = field(default_factory=list)
    h2d_ms: Optional[float] = None
    asynchronous: bool = False
    compute_done_event: Optional[Any] = None
    source_lease_events: List[Any] = field(default_factory=list)
    keepalive: List[torch.Tensor] = field(default_factory=list)
    action_materialization: Optional[Dict[str, Any]] = None
    stats_deferred: bool = False
    stats_accounted: bool = False
    timing_stats_accounted: bool = False
    integrity_samples: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = field(
        default_factory=dict
    )


class KTGroupExpertBufferManager:
    """Own and atomically publish a bounded pair of expert buffers."""

    WEIGHT_NAMES = ("w13_weight", "w2_weight")

    def __init__(self, config: KTGroupBufferConfig, *, strict_runtime: bool = True):
        self.config = config
        self.strict_runtime = strict_runtime
        self.layers: Dict[int, torch.nn.Module] = {}
        self.wrappers: Dict[int, Any] = {}
        self.initial_plans: Dict[int, Tuple[int, ...]] = {}
        self.weight_buffers: Dict[str, torch.Tensor] = {}
        self.records = [GroupBufferRecord(i) for i in range(config.buffer_count)]
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.copy_stream: Optional[Any] = None
        self.step_id = -1
        self.step_active = False
        self.active_group_id: Optional[int] = None
        self.active_buffer_id: Optional[int] = None
        self.active_is_fallback = False
        self.current_plans: Dict[int, Tuple[int, ...]] = {}
        self.loader: Optional[Any] = None
        self.provider: Optional[Any] = None
        self.inflight_keepalive: List[Tuple[Any, List[torch.Tensor]]] = []
        self._last_finished_group = -1
        self._action_group_id: Optional[int] = None
        self._action_ticket_id: Optional[int] = None
        self._action_plan_hash: Optional[str] = None
        self._action_materialization: Optional[Dict[str, Any]] = None
        self._action_logical_source_buffer_id: Optional[int] = None
        self._action_logical_source_plan: Optional[Dict[int, Tuple[int, ...]]] = None
        self._last_finished_buffer_id: Optional[int] = None
        self._pipeline_executor: Optional[ThreadPoolExecutor] = None
        self._pending_action: Optional[PendingGroupAction] = None
        self._pending_lock = threading.Lock()
        self._adopted_prefetch_record: Optional[GroupBufferRecord] = None
        self._adopted_prefetch_zero_load = False
        self._content_version = 0
        self._registration_logged = False
        self._integrity_check_enabled = (
            os.environ.get("SGLANG_KT_GROUP_INTEGRITY_CHECK", "0") == "1"
        )
        try:
            self._integrity_sample_width = int(
                os.environ.get("SGLANG_KT_GROUP_INTEGRITY_SAMPLE_WIDTH", "8")
            )
        except ValueError:
            self._integrity_sample_width = 8
        self._integrity_sample_width = max(1, min(64, self._integrity_sample_width))
        self.stats: Dict[str, float] = {
            "steps": 0,
            "actions": 0,
            "groups_executed": 0,
            "sync_loads": 0,
            "async_loads": 0,
            "commits": 0,
            "ready_hits": 0,
            "ready_misses": 0,
            "host_prefetch_experts_submitted": 0,
            "prefetch_deferred_buffer_busy": 0,
            "buffer_reclaim_block_ms": 0.0,
            "block_count": 0,
            "block_ms": 0.0,
            "cpu_fallback_groups": 0,
            "host_prepare_ms": 0.0,
            "prefetch_wait_ms": 0.0,
            "h2d_bytes": 0,
            "h2d_ms": 0.0,
            "overlap_ms": 0.0,
            "uncovered_boundary_tail_ms": 0.0,
            "oracle_hits": 0,
            "oracle_misses": 0,
            "active_overwrite_rejections": 0,
            "integrity_checks": 0,
            "full_loads": 0,
            "delta_loads": 0,
            "zero_loads": 0,
            "retain_experts": 0,
            "d2d_experts": 0,
            "h2d_experts": 0,
            "retained_bytes": 0,
            "d2d_bytes": 0,
            "pipeline_prefetch_submitted": 0,
            "pipeline_prefetch_adopted": 0,
            "pipeline_prefetch_zero_loads": 0,
            "pipeline_prefetch_mismatches": 0,
            "pipeline_prefetch_failures": 0,
            "pipeline_enqueue_tail_ms": 0.0,
            "pipeline_end_action_nonblocking": 0,
            "pipeline_sync_hints": 0,
        }
        self._log(
            "manager_created",
            physical_slots=config.physical_slots,
            group_size=config.group_size,
            slots_per_layer=config.slots_per_layer,
            buffer_count=config.buffer_count,
            load_mode=config.load_mode,
            miss_policy=config.miss_policy,
            prefetch_policy=config.prefetch_policy,
            integrity_check=self._integrity_check_enabled,
            materialization=config.materialization,
            max_replacements=config.max_replacements,
        )

    @property
    def physical_slots(self) -> int:
        return self.config.physical_slots

    @property
    def allocated_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.weight_buffers.values())

    @property
    def has_pending_action(self) -> bool:
        with self._pending_lock:
            return self._pending_action is not None

    def _log(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "step": self.step_id,
            "time": time.perf_counter(),
            **fields,
        }
        logger.info("[kt-group] %s", json.dumps(payload, sort_keys=True))

    @staticmethod
    def _storage_ptr(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().data_ptr())

    def _allocate_storage_from_layer(self, layer: torch.nn.Module) -> None:
        for name in self.WEIGHT_NAMES:
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError(f"group buffer requires Parameter {name}")
            if int(param.shape[0]) != self.config.slots_per_layer:
                raise ValueError(
                    f"{name} first dimension must equal slots_per_layer="
                    f"{self.config.slots_per_layer}, got {tuple(param.shape)}"
                )
            shape = (
                self.config.buffer_count,
                self.config.group_size,
                *tuple(param.shape),
            )
            self.weight_buffers[name] = torch.empty(
                shape, dtype=param.dtype, device=param.device
            )
        first = getattr(layer, self.WEIGHT_NAMES[0])
        self.device = first.device
        self.dtype = first.dtype
        if self.config.require_bf16 and self.dtype != torch.bfloat16:
            raise TypeError(
                f"M20-A supports BF16 expert buffers only, got {self.dtype}"
            )
        if self.device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device=self.device)
        self._log(
            "physical_storage_allocated",
            physical_slots=self.physical_slots,
            allocated_bytes=self.allocated_bytes,
            device=str(self.device),
            dtype=str(self.dtype),
            shapes={name: list(t.shape) for name, t in self.weight_buffers.items()},
        )

    def register_layer(self, layer_idx: int, layer: torch.nn.Module, wrapper: Any) -> None:
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.config.num_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")
        if layer_idx in self.layers:
            if self.layers[layer_idx] is not layer or self.wrappers[layer_idx] is not wrapper:
                raise RuntimeError(f"layer {layer_idx} registered twice with different objects")
            return
        if int(getattr(wrapper, "num_gpu_experts", -1)) != self.config.slots_per_layer:
            raise ValueError(
                f"layer {layer_idx} has num_gpu_experts="
                f"{getattr(wrapper, 'num_gpu_experts', None)}, expected "
                f"{self.config.slots_per_layer}"
            )
        if self.strict_runtime:
            method_name = type(getattr(wrapper, "gpu_method", None)).__name__
            if method_name != "UnquantizedFusedMoEMethod":
                raise TypeError(
                    "M20-A group buffers support the BF16 unquantized FusedMoE "
                    f"path only, got {method_name}"
                )
            if any(
                hasattr(layer, name)
                for name in ("w13_weight_scale", "w13_weight_scale_inv", "w13_weight_packed")
            ):
                raise TypeError("quantized expert layouts are not supported in M20-A")
        if not self.weight_buffers:
            self._allocate_storage_from_layer(layer)
        else:
            for name in self.WEIGHT_NAMES:
                param = getattr(layer, name)
                expected = tuple(self.weight_buffers[name].shape[2:])
                if tuple(param.shape) != expected:
                    raise ValueError(
                        f"layer {layer_idx} {name} shape {tuple(param.shape)} "
                        f"does not match shared shape {expected}"
                    )
                if param.device != self.device or param.dtype != self.dtype:
                    raise ValueError(
                        f"layer {layer_idx} {name} device/dtype mismatch: "
                        f"{param.device}/{param.dtype} vs {self.device}/{self.dtype}"
                    )

        selected = tuple(int(x) for x in wrapper.gpu_index_to_logical.tolist())
        if len(selected) != self.config.slots_per_layer:
            raise ValueError(
                f"layer {layer_idx} initial plan has {len(selected)} experts, "
                f"expected {self.config.slots_per_layer}"
            )
        self.layers[layer_idx] = layer
        self.wrappers[layer_idx] = wrapper
        self.initial_plans[layer_idx] = selected

        group_id = layer_idx // self.config.group_size
        offset = layer_idx % self.config.group_size
        self._bind_weight_views(layer_idx, group_id % self.config.buffer_count, offset)
        self._log(
            "layer_registered",
            layer_id=layer_idx,
            initial_buffer=group_id % self.config.buffer_count,
            layer_offset=offset,
            registered_layers=len(self.layers),
        )

    def _bind_weight_views(self, layer_idx: int, buffer_id: int, offset: int) -> None:
        layer = self.layers[layer_idx]
        for name in self.WEIGHT_NAMES:
            param = getattr(layer, name)
            view = self.weight_buffers[name][buffer_id, offset]
            if tuple(param.shape) != tuple(view.shape):
                raise RuntimeError(
                    f"cannot bind {name}: {tuple(param.shape)} != {tuple(view.shape)}"
                )
            param.data = view
            if self._storage_ptr(param) != self._storage_ptr(self.weight_buffers[name]):
                raise RuntimeError(f"{name} did not bind to manager-owned storage")

    def assert_registration_complete(self) -> None:
        missing = sorted(set(range(self.config.num_layers)) - set(self.layers))
        if missing:
            raise RuntimeError(f"group manager is missing layers: {missing}")
        if set(self.weight_buffers) != set(self.WEIGHT_NAMES):
            raise RuntimeError("group manager physical storage is incomplete")
        if not self._registration_logged:
            unique_storage_counts = {
                name: len(
                    {
                        self._storage_ptr(getattr(layer, name))
                        for layer in self.layers.values()
                    }
                )
                for name in self.WEIGHT_NAMES
            }
            if any(count != 1 for count in unique_storage_counts.values()):
                raise RuntimeError(
                    "layer Parameters do not share the manager-owned storage: "
                    f"{unique_storage_counts}"
                )
            self._registration_logged = True
            self._log(
                "registration_complete",
                registered_layers=len(self.layers),
                physical_slots=self.physical_slots,
                allocated_bytes=self.allocated_bytes,
                unique_storage_counts=unique_storage_counts,
            )

    def _normalize_plan(self, layer_idx: int, values: Sequence[int]) -> Tuple[int, ...]:
        selected: List[int] = []
        seen = set()
        for value in values:
            expert_id = int(value)
            if not 0 <= expert_id < self.config.num_experts or expert_id in seen:
                continue
            selected.append(expert_id)
            seen.add(expert_id)
            if len(selected) == self.config.slots_per_layer:
                break
        for expert_id in self.initial_plans[layer_idx]:
            if len(selected) == self.config.slots_per_layer:
                break
            if expert_id not in seen:
                selected.append(expert_id)
                seen.add(expert_id)
        if len(selected) != self.config.slots_per_layer:
            raise RuntimeError(
                f"layer {layer_idx} produced {len(selected)} unique experts, expected "
                f"{self.config.slots_per_layer}"
            )
        return tuple(selected)

    def _build_plans_for_layers(
        self, layer_indices: Sequence[int]
    ) -> Dict[int, Tuple[int, ...]]:
        plans: Dict[int, Tuple[int, ...]] = {}
        for layer_idx in layer_indices:
            selected: Optional[Sequence[int]] = None
            if self.config.prefetch_policy == "oracle":
                if self.provider is not None:
                    selected = self.provider.ranked_experts_for_active_step(
                        target_layer=layer_idx,
                        limit=self.config.slots_per_layer,
                    )
                if selected is None:
                    self.stats["oracle_misses"] += 1
                    if self.config.oracle_required:
                        raise RuntimeError(
                            "exact oracle metadata miss for "
                            f"step={self.step_id} layer={layer_idx}"
                        )
                else:
                    self.stats["oracle_hits"] += 1
            if selected is None:
                selected = self.initial_plans[layer_idx]
            plans[layer_idx] = self._normalize_plan(layer_idx, selected)
        return plans

    def _build_step_plans(self) -> Dict[int, Tuple[int, ...]]:
        return self._build_plans_for_layers(range(self.config.num_layers))

    @staticmethod
    def _plan_hash(group_id: int, plans: Dict[int, Tuple[int, ...]]) -> str:
        payload = {
            "group_id": int(group_id),
            "layer_plans": [
                [int(layer_idx), list(plans[layer_idx])]
                for layer_idx in sorted(plans)
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _plans_tuple(
        plans: Optional[Dict[int, Tuple[int, ...]]],
    ) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
        if plans is None:
            return ()
        return tuple(
            (int(layer_idx), tuple(int(value) for value in plans[layer_idx]))
            for layer_idx in sorted(plans)
        )

    def _current_action_spec(self) -> GroupActionSpec:
        if self._action_group_id is None or self._action_plan_hash is None:
            raise RuntimeError("no stage action is active")
        return GroupActionSpec(
            group_id=int(self._action_group_id),
            plans=self._plans_tuple(self.current_plans),
            ticket_id=self._action_ticket_id,
            logical_source_plans=self._plans_tuple(
                self._action_logical_source_plan
            ),
            logical_source_buffer_id=self._action_logical_source_buffer_id,
            plan_hash=self._action_plan_hash,
        )

    @staticmethod
    def _action_specs_match(left: GroupActionSpec, right: GroupActionSpec) -> bool:
        return (
            left.ticket_id == right.ticket_id
            and left.group_id == right.group_id
            and left.plan_hash == right.plan_hash
            and left.plans == right.plans
        )

    def _validate_async_loader(self, loader: Any) -> None:
        if (
            self.config.load_mode == "async"
            and self.copy_stream is not None
            and not bool(getattr(loader, "group_async_host_memory_enabled", False))
        ):
            raise RuntimeError(
                "async group H2D requires page-locked host memory; enable "
                "SGLANG_KT_RUNTIME_PIN_CPU_TENSORS=1 or host registration"
            )

    def _expert_nbytes(self, buffer_id: int, offset: int, slot: int) -> int:
        return sum(
            int(self.weight_buffers[name][buffer_id, offset, slot].numel())
            * int(self.weight_buffers[name][buffer_id, offset, slot].element_size())
            for name in self.WEIGHT_NAMES
        )

    @staticmethod
    def _record_slot(
        record: GroupBufferRecord, layer_idx: int, expert_id: int
    ) -> Optional[int]:
        experts = record.slot_experts.get(int(layer_idx), ())
        valid = record.valid_slots.get(int(layer_idx), ())
        for slot, value in enumerate(experts):
            if slot < len(valid) and valid[slot] and value == int(expert_id):
                return slot
        return None

    def _record_has_exact_plan(
        self,
        record: GroupBufferRecord,
        group_id: int,
        plans: Dict[int, Tuple[int, ...]],
    ) -> bool:
        if record.group_id != group_id or record.plan != plans:
            return False
        start, end = self.config.group_range(group_id)
        for layer_idx in range(start, end):
            valid = record.valid_slots.get(layer_idx, ())
            if (
                record.slot_experts.get(layer_idx) != plans[layer_idx]
                or len(valid) != self.config.slots_per_layer
                or not all(valid)
            ):
                return False
        return True

    def _records_for_group(self, group_id: int) -> List[GroupBufferRecord]:
        candidates = [
            record
            for record in self.records
            if record.group_id == int(group_id)
            and record.state in {GroupBufferState.READY, GroupBufferState.REUSABLE}
            and all(
                len(record.valid_slots.get(layer_idx, ()))
                == self.config.slots_per_layer
                and all(record.valid_slots[layer_idx])
                for layer_idx in range(*self.config.group_range(int(group_id)))
            )
        ]
        return sorted(
            candidates,
            key=lambda record: (
                record.buffer_id == self._last_finished_buffer_id,
                record.content_version,
                record.version,
            ),
            reverse=True,
        )

    def _latest_record_for_group(
        self, group_id: int
    ) -> Optional[GroupBufferRecord]:
        candidates = self._records_for_group(group_id)
        return candidates[0] if candidates else None

    def _bound_plans_to_source(
        self,
        group_id: int,
        desired: Dict[int, Tuple[int, ...]],
        source: Optional[GroupBufferRecord],
        *,
        explicit: bool,
        logical_source_plan: Optional[Dict[int, Tuple[int, ...]]] = None,
    ) -> Dict[int, Tuple[int, ...]]:
        """Apply or validate the per-layer replacement budget.

        A cold action has no active same-stage plan and therefore fills all S
        slots.  Subsequent generated plans retain the source slot order and
        replace at most K victims.  Replay plans are immutable and fail closed
        if they violate the same budget.
        """

        # K constrains the logical placement transition, independently of how
        # that placement is materialized.  The paired full-load reference must
        # build the same bounded plan as delta replay; only its copy operations
        # are allowed to differ.
        if source is None and logical_source_plan is None:
            return desired
        max_replacements = int(self.config.max_replacements or 0)
        if max_replacements == self.config.slots_per_layer:
            return desired
        bounded: Dict[int, Tuple[int, ...]] = {}
        start, end = self.config.group_range(group_id)
        for layer_idx in range(start, end):
            active = tuple(
                logical_source_plan[layer_idx]
                if logical_source_plan is not None
                else source.plan[layer_idx]
            )
            target = tuple(desired[layer_idx])
            changed = len(set(target) - set(active))
            if explicit:
                if changed > max_replacements:
                    raise RuntimeError(
                        "explicit action plan exceeds replacement budget: "
                        f"group={group_id} layer={layer_idx} changed={changed} "
                        f"K={max_replacements}"
                    )
                bounded[layer_idx] = target
                continue

            new_experts = [expert for expert in target if expert not in active]
            accepted = new_experts[:max_replacements]
            accepted_set = set(accepted)
            victims = [
                expert
                for expert in reversed(active)
                if expert not in target
            ][: len(accepted)]
            victim_to_new = dict(zip(victims, accepted))
            result = tuple(victim_to_new.get(expert, expert) for expert in active)
            if len(result) != self.config.slots_per_layer or len(set(result)) != len(result):
                raise RuntimeError(
                    f"bounded plan construction failed for layer {layer_idx}: {result}"
                )
            if not accepted_set.issubset(result):
                raise RuntimeError(
                    f"bounded plan lost accepted experts for layer {layer_idx}"
                )
            target_record = self.records[1 - source.buffer_id]
            if target_record.group_id == group_id:
                result_set = set(result)
                assigned: List[Optional[int]] = [None] * len(result)
                used = set()
                for slot, expert in enumerate(
                    target_record.slot_experts.get(layer_idx, ())
                ):
                    valid = target_record.valid_slots.get(layer_idx, ())
                    if (
                        slot < len(assigned)
                        and slot < len(valid)
                        and valid[slot]
                        and expert in result_set
                    ):
                        assigned[slot] = expert
                        used.add(expert)
                remaining = [expert for expert in result if expert not in used]
                iterator = iter(remaining)
                result = tuple(
                    next(iterator) if expert is None else expert
                    for expert in assigned
                )
            bounded[layer_idx] = result
        return bounded

    def plan_materialization(
        self,
        group_id: int,
        target_buffer_id: int,
        *,
        source_buffer_id: Optional[int] = None,
        plans: Optional[Dict[int, Tuple[int, ...]]] = None,
    ) -> Tuple[GroupCopyOp, ...]:
        """Resolve target slots to retain, same-group D2D, or host H2D."""

        group_id = int(group_id)
        target_buffer_id = int(target_buffer_id)
        target = self.records[target_buffer_id]
        source = (
            None
            if source_buffer_id is None
            else self.records[int(source_buffer_id)]
        )
        selected_plans = self.current_plans if plans is None else plans
        start, end = self.config.group_range(group_id)
        ops: List[GroupCopyOp] = []
        for layer_idx in range(start, end):
            offset = layer_idx - start
            for dst_slot, expert_id in enumerate(selected_plans[layer_idx]):
                nbytes = self._expert_nbytes(target_buffer_id, offset, dst_slot)
                target_slot = self._record_slot(target, layer_idx, expert_id)
                if target.group_id == group_id and target_slot == dst_slot:
                    ops.append(
                        GroupCopyOp(
                            kind=GroupCopyKind.RETAIN,
                            layer_idx=layer_idx,
                            expert_id=expert_id,
                            dst_buffer_id=target_buffer_id,
                            dst_slot=dst_slot,
                            nbytes=nbytes,
                            src_buffer_id=target_buffer_id,
                            src_slot=dst_slot,
                        )
                    )
                    continue
                source_slot = (
                    None
                    if source is None or source.group_id != group_id
                    else self._record_slot(source, layer_idx, expert_id)
                )
                if source_slot is not None and source_buffer_id != target_buffer_id:
                    ops.append(
                        GroupCopyOp(
                            kind=GroupCopyKind.D2D,
                            layer_idx=layer_idx,
                            expert_id=expert_id,
                            dst_buffer_id=target_buffer_id,
                            dst_slot=dst_slot,
                            nbytes=nbytes,
                            src_buffer_id=int(source_buffer_id),
                            src_slot=source_slot,
                        )
                    )
                else:
                    ops.append(
                        GroupCopyOp(
                            kind=GroupCopyKind.H2D,
                            layer_idx=layer_idx,
                            expert_id=expert_id,
                            dst_buffer_id=target_buffer_id,
                            dst_slot=dst_slot,
                            nbytes=nbytes,
                        )
                    )
        return tuple(ops)

    def begin_step(self, *, num_tokens: int, loader: Any, provider: Optional[Any]) -> None:
        self.assert_registration_complete()
        if self.step_active or self.active_group_id is not None:
            raise RuntimeError("cannot begin a group step while another step is active")
        self._validate_async_loader(loader)
        self.step_id += 1
        self.step_active = True
        self._action_group_id = None
        self._last_finished_group = -1
        self.loader = loader
        self.provider = provider
        if self.config.prefetch_policy == "oracle":
            if provider is None and self.config.oracle_required:
                raise RuntimeError("group-buffer oracle policy requires an oracle provider")
            if provider is not None:
                provider.begin_step_if_layer0(0, int(num_tokens))
        self.current_plans = self._build_step_plans()
        self.stats["steps"] += 1
        self._log(
            "step_begin",
            num_tokens=int(num_tokens),
            groups=self.config.num_groups,
            oracle_hits=int(self.stats["oracle_hits"]),
            oracle_misses=int(self.stats["oracle_misses"]),
        )

    def begin_action(
        self,
        *,
        group_id: int,
        num_tokens: int,
        loader: Any,
        provider: Optional[Any],
        plans: Optional[Dict[int, Sequence[int]]] = None,
        ticket_id: Optional[int] = None,
        logical_source_plans: Optional[Dict[int, Sequence[int]]] = None,
        logical_source_buffer_id: Optional[int] = None,
        plan_hash: Optional[str] = None,
    ) -> Dict[int, Tuple[int, ...]]:
        """Start one independently scheduled stage action.

        Every action derives its plan from the currently packed cohort metadata
        or consumes an explicit ticket plan. This keeps interleaved cohorts
        independent of provider-global active-step state.
        """

        self.assert_registration_complete()
        group_id = int(group_id)
        self.config.group_range(group_id)
        if self.step_active or self.active_group_id is not None:
            raise RuntimeError("cannot begin an action while another action is active")
        self._validate_async_loader(loader)
        self.step_id += 1
        self.step_active = True
        self._action_group_id = group_id
        self._action_ticket_id = None if ticket_id is None else int(ticket_id)
        self._last_finished_group = group_id - 1
        self.loader = loader
        self.provider = provider
        self._adopted_prefetch_record = None
        self._adopted_prefetch_zero_load = False

        explicit_plans = plans is not None
        if plans is None:
            if self.config.prefetch_policy == "oracle":
                if provider is None and self.config.oracle_required:
                    raise RuntimeError(
                        "group-buffer oracle policy requires an oracle provider"
                    )
                if provider is not None:
                    provider.begin_step_if_layer0(0, int(num_tokens))
            start, end = self.config.group_range(group_id)
            self.current_plans = self._build_plans_for_layers(range(start, end))
        else:
            start, end = self.config.group_range(group_id)
            expected_layers = set(range(start, end))
            missing = sorted(expected_layers - set(plans))
            extra = sorted(set(plans) - expected_layers)
            if missing or extra:
                raise RuntimeError(
                    "action plan snapshot layer mismatch: "
                    f"missing={missing} extra={extra}"
                )
            self.current_plans = {
                layer_idx: self._normalize_plan(layer_idx, plans[layer_idx])
                for layer_idx in range(start, end)
            }

        source = self._latest_record_for_group(group_id)
        ticket_source_plan: Optional[Dict[int, Tuple[int, ...]]] = None
        if logical_source_plans is not None:
            expected_layers = set(range(start, end))
            missing = sorted(expected_layers - set(logical_source_plans))
            extra = sorted(set(logical_source_plans) - expected_layers)
            if missing or extra:
                raise RuntimeError(
                    "ticket logical source layer mismatch: "
                    f"missing={missing} extra={extra}"
                )
            ticket_source_plan = {}
            for layer_idx in range(start, end):
                raw = tuple(int(value) for value in logical_source_plans[layer_idx])
                normalized = self._normalize_plan(layer_idx, raw)
                if raw != normalized:
                    raise RuntimeError(
                        f"invalid ticket logical source plan for layer {layer_idx}"
                    )
                ticket_source_plan[layer_idx] = normalized
        if logical_source_buffer_id is not None and not 0 <= int(
            logical_source_buffer_id
        ) < self.config.buffer_count:
            raise RuntimeError("ticket logical source buffer ID is out of range")
        # A replay ticket's logical source is part of its frozen provenance.
        # In particular, ``None`` means the planner did not bind the action to
        # an abstract buffer, even when this replay happens to find a useful
        # physical cache record.  Keep that distinction so sync and async
        # replays report the same ticket metadata; physical materialization
        # still selects ``source`` independently below.
        if ticket_source_plan is not None:
            self._action_logical_source_buffer_id = (
                None
                if logical_source_buffer_id is None
                else int(logical_source_buffer_id)
            )
        else:
            self._action_logical_source_buffer_id = (
                None if source is None else int(source.buffer_id)
            )
        self._action_logical_source_plan = (
            ticket_source_plan
            if ticket_source_plan is not None
            else (None if source is None else dict(source.plan))
        )
        self.current_plans = self._bound_plans_to_source(
            group_id,
            self.current_plans,
            source,
            explicit=explicit_plans,
            logical_source_plan=ticket_source_plan,
        )

        self.stats["actions"] += 1
        frozen = dict(self.current_plans)
        actual_hash = self._plan_hash(group_id, frozen)
        if plan_hash is not None and str(plan_hash) != actual_hash:
            raise RuntimeError(
                f"action ticket plan hash mismatch: expected={plan_hash} actual={actual_hash}"
            )
        self._action_plan_hash = actual_hash
        self._action_materialization = None
        self._adopt_pending_action(self._current_action_spec())
        start, end = self.config.group_range(group_id)
        self._log(
            "action_begin",
            action_group_id=group_id,
            num_tokens=int(num_tokens),
            layers=[start, end],
            ticket_id=self._action_ticket_id,
            logical_source_buffer_id=self._action_logical_source_buffer_id,
            plan_hash=actual_hash,
        )
        return frozen

    def action_materialization_snapshot(self) -> Dict[str, Any]:
        if not self.step_active or self._action_group_id is None:
            raise RuntimeError("no stage action is active")
        if self._action_materialization is None:
            raise RuntimeError("stage action has not materialized its group plan")
        return dict(self._action_materialization)

    def _build_action_materialization(
        self,
        record: GroupBufferRecord,
        *,
        source_buffer_id: Optional[int],
        materialization: str,
        expected_buffer_versions: Sequence[int],
        spec: Optional[GroupActionSpec] = None,
        physical_source_matches_logical: Optional[bool] = None,
    ) -> Dict[str, Any]:
        physical_source = (
            None
            if source_buffer_id is None
            else self.records[int(source_buffer_id)]
        )
        active_spec = spec
        if active_spec is None and self._action_group_id is not None:
            active_spec = self._current_action_spec()
        logical_source_plan = (
            None
            if active_spec is None
            else active_spec.logical_source_plan_dict
        )
        if physical_source_matches_logical is None:
            physical_source_matches_logical = bool(
                logical_source_plan is not None
                and physical_source is not None
                and physical_source.group_id == record.group_id
                and physical_source.plan == logical_source_plan
            )
        changed_by_layer: Dict[str, int] = {}
        if logical_source_plan is not None:
            for layer_idx, target in record.plan.items():
                changed_by_layer[str(layer_idx)] = len(
                    set(target) - set(logical_source_plan[layer_idx])
                )
        else:
            changed_by_layer = {
                str(layer_idx): len(target)
                for layer_idx, target in record.plan.items()
            }
        max_changed = max(changed_by_layer.values(), default=0)
        if (
            logical_source_plan is not None
            and max_changed > int(self.config.max_replacements or 0)
        ):
            raise RuntimeError(
                "materialized action exceeded its replacement budget: "
                f"changed={max_changed} K={self.config.max_replacements}"
            )
        if (
            physical_source_matches_logical
            and materialization != "full"
            and record.h2d_experts > sum(changed_by_layer.values())
        ):
            raise RuntimeError(
                "delta H2D experts exceed logical replacements: "
                f"h2d={record.h2d_experts} changed={sum(changed_by_layer.values())}"
            )
        result = {
            "ticket_id": (
                None if active_spec is None else active_spec.ticket_id
            ),
            "group_id": record.group_id,
            "plan_hash": record.plan_hash,
            "materialization": str(materialization),
            "logical_source_buffer_id": (
                None
                if active_spec is None
                else active_spec.logical_source_buffer_id
            ),
            "logical_source_plan_present": logical_source_plan is not None,
            "logical_source_plan_hash": (
                None
                if logical_source_plan is None
                else self._plan_hash(record.group_id, logical_source_plan)
            ),
            "physical_source_matches_logical": bool(
                physical_source_matches_logical
            ),
            "source_buffer_id": source_buffer_id,
            "target_buffer_id": record.buffer_id,
            "expected_buffer_versions": [
                int(value) for value in expected_buffer_versions
            ],
            "buffer_versions": [int(item.version) for item in self.records],
            "copy_ops": list(record.copy_ops),
            "changed_by_layer": changed_by_layer,
            "max_changed": max_changed,
            "h2d_experts": int(record.h2d_experts),
            "d2d_experts": int(record.d2d_experts),
            "retained_experts": int(record.retained_experts),
            "h2d_bytes": int(record.h2d_bytes),
            "d2d_bytes": int(record.d2d_bytes),
            "retained_bytes": int(record.retained_bytes),
        }
        return result

    def _set_action_materialization(
        self,
        record: GroupBufferRecord,
        *,
        source_buffer_id: Optional[int],
        materialization: str,
        expected_buffer_versions: Sequence[int],
        spec: Optional[GroupActionSpec] = None,
        physical_source_matches_logical: Optional[bool] = None,
    ) -> None:
        self._action_materialization = self._build_action_materialization(
            record,
            source_buffer_id=source_buffer_id,
            materialization=materialization,
            expected_buffer_versions=expected_buffer_versions,
            spec=spec,
            physical_source_matches_logical=physical_source_matches_logical,
        )
        record.action_materialization = dict(self._action_materialization)

    def _prepare_expert(
        self,
        *,
        layer_idx: int,
        expert_id: int,
        buffer_id: int,
        offset: int,
        slot: int,
        loader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        active_loader = self.loader if loader is None else loader
        if active_loader is None:
            raise RuntimeError("group manager has no expert loader")
        prepare = getattr(active_loader, "prepare_expert_for_group_slot", None)
        if not callable(prepare):
            raise TypeError(
                "group expert loader must implement prepare_expert_for_group_slot"
            )
        return prepare(
            layer_idx=layer_idx,
            logical_expert_id=expert_id,
            w13_dst=self.weight_buffers["w13_weight"][buffer_id, offset, slot],
            w2_dst=self.weight_buffers["w2_weight"][buffer_id, offset, slot],
        )

    def _capture_integrity_sample(
        self,
        record: GroupBufferRecord,
        *,
        layer_idx: int,
        slot: int,
        prepared: Dict[str, Any],
    ) -> None:
        """Retain small host samples for an opt-in post-H2D equivalence check."""

        if not self._integrity_check_enabled or slot != 0:
            return
        anchors = {0, self.config.num_layers // 2, self.config.num_layers - 1}
        if layer_idx not in anchors:
            return
        tensors = prepared.get("tensors")
        if not isinstance(tensors, tuple) or len(tensors) != 3:
            raise TypeError("prepared group expert must contain three tensors")
        gate, up, down = tensors
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("prepared group expert contains a non-tensor value")
        record.integrity_samples[(layer_idx, slot)] = {
            name: tensor.detach()
            .reshape(-1)[: self._integrity_sample_width]
            .to(device="cpu")
            .clone()
            for name, tensor in (("gate", gate), ("up", up), ("down", down))
        }

    def _capture_d2d_integrity_sample(
        self, record: GroupBufferRecord, op: GroupCopyOp, *, start: int
    ) -> None:
        if not self._integrity_check_enabled or op.dst_slot != 0:
            return
        anchors = {0, self.config.num_layers // 2, self.config.num_layers - 1}
        if op.layer_idx not in anchors:
            return
        if op.src_buffer_id is None or op.src_slot is None:
            raise RuntimeError("D2D integrity sample has no source")
        offset = op.layer_idx - start
        w13 = self.weight_buffers["w13_weight"][
            op.src_buffer_id, offset, op.src_slot
        ]
        w2 = self.weight_buffers["w2_weight"][
            op.src_buffer_id, offset, op.src_slot
        ]
        shard = int(w13.shape[0]) // 2
        record.integrity_samples[(op.layer_idx, op.dst_slot)] = {
            name: tensor.detach()
            .reshape(-1)[: self._integrity_sample_width]
            .to(device="cpu")
            .clone()
            for name, tensor in (
                ("gate", w13[:shard]),
                ("up", w13[shard:]),
                ("down", w2),
            )
        }

    def _verify_record_integrity(self, record: GroupBufferRecord) -> None:
        """Verify sampled BF16 payloads after the record's H2D event completes."""

        if not self._integrity_check_enabled or not record.integrity_samples:
            return
        if record.group_id is None:
            raise RuntimeError("integrity check found a record without a group id")
        start, end = self.config.group_range(record.group_id)
        verified = 0
        for (layer_idx, slot), expected in record.integrity_samples.items():
            if not start <= layer_idx < end:
                raise RuntimeError(
                    f"integrity sample layer {layer_idx} is outside group {record.group_id}"
                )
            offset = layer_idx - start
            w13 = self.weight_buffers["w13_weight"][record.buffer_id, offset, slot]
            w2 = self.weight_buffers["w2_weight"][record.buffer_id, offset, slot]
            if w13.dim() < 2 or int(w13.shape[0]) % 2:
                raise RuntimeError(f"invalid w13 layout for integrity check: {tuple(w13.shape)}")
            shard = int(w13.shape[0]) // 2
            actual = {
                "gate": w13[:shard],
                "up": w13[shard:],
                "down": w2,
            }
            for name, source_sample in expected.items():
                destination_sample = (
                    actual[name]
                    .detach()
                    .reshape(-1)[: source_sample.numel()]
                    .to(device="cpu")
                )
                if not torch.equal(destination_sample, source_sample):
                    raise RuntimeError(
                        "group H2D integrity mismatch: "
                        f"group={record.group_id} layer={layer_idx} slot={slot} "
                        f"tensor={name}"
                    )
                verified += 1
        self.stats["integrity_checks"] += verified
        self._log(
            "group_integrity_verified",
            group_id=record.group_id,
            buffer_id=record.buffer_id,
            samples=verified,
        )

    def _verify_committed_mapping(self, layer_idx: int, selected: Sequence[int]) -> None:
        """Check all mapping replicas only in explicit diagnostic mode."""

        if not self._integrity_check_enabled:
            return
        wrapper = self.wrappers[layer_idx]
        expected_mask = torch.zeros(self.config.num_experts, dtype=torch.bool)
        expected_logical = torch.full(
            (self.config.num_experts,), -1, dtype=torch.int32
        )
        expected_reverse = torch.tensor(selected, dtype=torch.int32)
        if selected:
            ids = torch.tensor(selected, dtype=torch.long)
            expected_mask[ids] = True
            expected_logical[ids] = torch.arange(len(selected), dtype=torch.int32)
        replicas = {
            "mask": getattr(wrapper, "gpu_experts_mask", None),
            "logical": getattr(wrapper, "logical_to_gpu_index", None),
            "reverse": getattr(wrapper, "gpu_index_to_logical", None),
            "cuda_mask": getattr(wrapper, "gpu_experts_mask_cuda", None),
            "cuda_logical": getattr(wrapper, "logical_to_gpu_index_cuda", None),
        }
        expected = {
            "mask": expected_mask,
            "logical": expected_logical,
            "reverse": expected_reverse,
            "cuda_mask": expected_mask,
            "cuda_logical": expected_logical,
        }
        for name, value in replicas.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or not torch.equal(
                value.to(device="cpu"), expected[name]
            ):
                raise RuntimeError(
                    f"group mapping integrity mismatch: layer={layer_idx} replica={name}"
                )
        native = getattr(wrapper, "wrapper", None)
        native_mask = getattr(native, "gpu_experts_mask", None)
        if isinstance(native_mask, torch.Tensor) and not torch.equal(
            native_mask.to(device="cpu"), expected_mask
        ):
            raise RuntimeError(f"group mapping integrity mismatch: layer={layer_idx} native_mask")

    def _enqueue_prepared_expert(
        self,
        *,
        prepared: Dict[str, Any],
        buffer_id: int,
        offset: int,
        slot: int,
        loader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        active_loader = self.loader if loader is None else loader
        if active_loader is None:
            raise RuntimeError("group manager has no expert loader")
        enqueue = getattr(active_loader, "enqueue_prepared_group_expert", None)
        if not callable(enqueue):
            raise TypeError(
                "group expert loader must implement enqueue_prepared_group_expert"
            )
        return enqueue(
            prepared=prepared,
            w13_dst=self.weight_buffers["w13_weight"][buffer_id, offset, slot],
            w2_dst=self.weight_buffers["w2_weight"][buffer_id, offset, slot],
            stream=self.copy_stream,
        )

    def _drain_inflight_keepalive(self, *, block: bool = False) -> None:
        retained: List[Tuple[Any, List[torch.Tensor]]] = []
        for event, tensors in self.inflight_keepalive:
            if block:
                event.synchronize()
                continue
            if not event.query():
                retained.append((event, tensors))
        self.inflight_keepalive = retained

    @staticmethod
    def _drain_source_leases(
        record: GroupBufferRecord, *, block: bool
    ) -> bool:
        retained = []
        for event in record.source_lease_events:
            if block:
                event.synchronize()
            elif not event.query():
                retained.append(event)
        record.source_lease_events = retained
        return not retained

    def _account_record_stats(self, record: GroupBufferRecord) -> None:
        """Attribute a materialization to the action that adopts it."""

        if record.stats_accounted:
            return
        self.stats["host_prepare_ms"] += record.host_prepare_ms
        self.stats["prefetch_wait_ms"] += record.prefetch_wait_ms
        self.stats["h2d_bytes"] += record.h2d_bytes
        self.stats["d2d_bytes"] += record.d2d_bytes
        self.stats["retained_bytes"] += record.retained_bytes
        self.stats["h2d_experts"] += record.h2d_experts
        self.stats["d2d_experts"] += record.d2d_experts
        self.stats["retain_experts"] += record.retained_experts
        self.stats[
            "delta_loads" if record.materialization_kind == "delta" else "full_loads"
        ] += 1
        self.stats["async_loads" if record.asynchronous else "sync_loads"] += 1
        record.stats_accounted = True
        self._account_record_timing(record)

    def _account_record_timing(self, record: GroupBufferRecord) -> None:
        if record.timing_stats_accounted or record.h2d_ms is None:
            return
        self.stats["h2d_ms"] += record.h2d_ms
        record.timing_stats_accounted = True

    def _invalidate_record(self, record: GroupBufferRecord) -> None:
        """Drop a failed lookahead record without exposing partial weights."""

        record.state = GroupBufferState.EMPTY
        record.group_id = None
        record.plan = {}
        record.plan_hash = None
        record.ticket_id = None
        record.slot_experts = {}
        record.valid_slots = {}
        record.ready_event = None
        record.timing_start_event = None
        record.timing_end_event = None
        record.compute_done_event = None
        record.action_materialization = None
        record.keepalive.clear()
        record.integrity_samples.clear()
        record.source_lease_events.clear()
        record.stats_deferred = False
        record.stats_accounted = False
        record.timing_stats_accounted = False

    def _enqueue_group_load(
        self,
        group_id: int,
        buffer_id: int,
        *,
        asynchronous: bool,
        source_buffer_id: Optional[int] = None,
        action_spec: Optional[GroupActionSpec] = None,
        loader: Optional[Any] = None,
    ) -> GroupBufferRecord:
        active_loader = self.loader if loader is None else loader
        if active_loader is None:
            raise RuntimeError("group manager has no expert loader")
        if self.active_buffer_id == buffer_id:
            self.stats["active_overwrite_rejections"] += 1
            raise RuntimeError(
                f"refusing to overwrite ACTIVE buffer {buffer_id} for group {group_id}"
            )
        self._drain_inflight_keepalive()
        record = self.records[buffer_id]
        previous_group_id = record.group_id
        previous_compute_done_event = record.compute_done_event
        if not self._drain_source_leases(record, block=False):
            t0 = time.perf_counter()
            self._drain_source_leases(record, block=True)
            lease_wait_ms = (time.perf_counter() - t0) * 1000.0
            self.stats["buffer_reclaim_block_ms"] += lease_wait_ms
            self._log(
                "group_buffer_wait_source_leases",
                buffer_id=buffer_id,
                previous_group_id=previous_group_id,
                wait_ms=lease_wait_ms,
            )
        if record.state == GroupBufferState.LOADING:
            # The destination storage is still being written by the old H2D.
            # Retaining source tensors alone is insufficient: overwriting this
            # buffer before its event completes is a write/write race.
            t0 = time.perf_counter()
            self._mark_ready(record, block=True)
            reclaim_ms = (time.perf_counter() - t0) * 1000.0
            self.stats["buffer_reclaim_block_ms"] += reclaim_ms
            self._log(
                "group_buffer_reclaimed",
                buffer_id=buffer_id,
                previous_group_id=previous_group_id,
                reclaim_ms=reclaim_ms,
            )
        expected_buffer_versions = tuple(record.version for record in self.records)
        selected_plans = (
            self.current_plans
            if action_spec is None
            else action_spec.plan_dict
        )
        if source_buffer_id is not None:
            source_buffer_id = int(source_buffer_id)
            if not 0 <= source_buffer_id < self.config.buffer_count:
                raise IndexError(f"source buffer out of range: {source_buffer_id}")
        elif self.config.materialization == "delta":
            source = self._latest_record_for_group(group_id)
            if source is not None and source.buffer_id != buffer_id:
                source_buffer_id = source.buffer_id
        active_spec = action_spec
        if active_spec is None and self._action_group_id is not None:
            active_spec = self._current_action_spec()
        logical_source_plan = (
            None
            if active_spec is None
            else active_spec.logical_source_plan_dict
        )
        physical_source = (
            None
            if source_buffer_id is None
            else self.records[source_buffer_id]
        )
        # Freeze this relation before target-buffer metadata is overwritten.
        # The physical cache can legitimately hold an older plan than the
        # ticket's logical source after A/B eviction by another group.
        physical_source_matches_logical = bool(
            logical_source_plan is not None
            and physical_source is not None
            and physical_source.group_id == group_id
            and physical_source.plan == logical_source_plan
        )
        if self.config.materialization == "delta":
            copy_ops = self.plan_materialization(
                group_id,
                buffer_id,
                source_buffer_id=source_buffer_id,
                plans=selected_plans,
            )
            materialization_kind = "delta"
        else:
            start, end = self.config.group_range(group_id)
            copy_ops = tuple(
                GroupCopyOp(
                    kind=GroupCopyKind.H2D,
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                    dst_buffer_id=buffer_id,
                    dst_slot=slot,
                    nbytes=self._expert_nbytes(
                        buffer_id, layer_idx - start, slot
                    ),
                )
                for layer_idx in range(start, end)
                for slot, expert_id in enumerate(selected_plans[layer_idx])
            )
            materialization_kind = "full"
        record.keepalive.clear()
        record.integrity_samples.clear()
        record.version += 1
        record.group_id = group_id
        record.state = GroupBufferState.LOADING
        start, end = self.config.group_range(group_id)
        record.plan = {layer: selected_plans[layer] for layer in range(start, end)}
        record.plan_hash = self._plan_hash(group_id, record.plan)
        record.ticket_id = (
            self._action_ticket_id
            if action_spec is None
            else action_spec.ticket_id
        )
        record.slot_experts = {
            layer_idx: tuple(record.plan[layer_idx])
            for layer_idx in range(start, end)
        }
        record.valid_slots = {
            layer_idx: (False,) * self.config.slots_per_layer
            for layer_idx in range(start, end)
        }
        record.load_started_at = time.perf_counter()
        record.host_prepare_completed_at = None
        record.load_enqueued_at = None
        record.load_completed_at = None
        record.host_prepare_ms = 0.0
        record.prefetch_wait_ms = 0.0
        record.h2d_bytes = 0
        record.d2d_bytes = sum(
            op.nbytes for op in copy_ops if op.kind == GroupCopyKind.D2D
        )
        record.retained_bytes = sum(
            op.nbytes for op in copy_ops if op.kind == GroupCopyKind.RETAIN
        )
        record.h2d_experts = sum(
            op.kind == GroupCopyKind.H2D for op in copy_ops
        )
        record.d2d_experts = sum(
            op.kind == GroupCopyKind.D2D for op in copy_ops
        )
        record.retained_experts = sum(
            op.kind == GroupCopyKind.RETAIN for op in copy_ops
        )
        record.materialization_kind = materialization_kind
        record.copy_ops = [op.to_dict() for op in copy_ops]
        record.h2d_ms = None
        record.asynchronous = bool(asynchronous)
        record.keepalive = []
        record.action_materialization = None
        record.stats_deferred = bool(action_spec is not None)
        record.stats_accounted = False
        record.timing_stats_accounted = False

        prepared_items: List[Tuple[int, int, Dict[str, Any]]] = []
        for op in copy_ops:
            if op.kind == GroupCopyKind.H2D:
                layer_idx = op.layer_idx
                offset = layer_idx - start
                slot = op.dst_slot
                expert_id = op.expert_id
                info = self._prepare_expert(
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                    buffer_id=buffer_id,
                    offset=offset,
                    slot=slot,
                    loader=active_loader,
                )
                if not isinstance(info, dict):
                    raise TypeError("prepared group expert metadata must be a dict")
                record.h2d_bytes += int(info.get("h2d_bytes", 0) or 0)
                record.prefetch_wait_ms += float(
                    info.get("prefetch_wait_ms", 0.0) or 0.0
                )
                if (
                    asynchronous
                    and self.copy_stream is not None
                    and not bool(info.get("async_h2d_capable", False))
                ):
                    raise RuntimeError(
                        "async group H2D requires pinned or CUDA-registered host "
                        "expert tensors; enable SGLANG_KT_RUNTIME_PIN_CPU_TENSORS=1"
                    )
                prepared_items.append((offset, slot, info))
                self._capture_integrity_sample(
                    record,
                    layer_idx=layer_idx,
                    slot=slot,
                    prepared=info,
                )

        record.host_prepare_completed_at = time.perf_counter()
        record.host_prepare_ms = (
            record.host_prepare_completed_at - record.load_started_at
        ) * 1000.0

        if self.copy_stream is not None:
            if previous_compute_done_event is not None:
                # This storage may still be read by an earlier group on the
                # serving stream.  The wait is queued on the copy stream, so
                # CPU preparation above remains concurrent with that work.
                self.copy_stream.wait_event(previous_compute_done_event)
                self._log(
                    "group_buffer_wait_previous_compute",
                    buffer_id=buffer_id,
                    previous_group_id=previous_group_id,
                )
            if source_buffer_id is not None:
                source_compute_done = self.records[source_buffer_id].compute_done_event
                if source_compute_done is not None:
                    self.copy_stream.wait_event(source_compute_done)
            record.compute_done_event = None
            record.timing_start_event = torch.cuda.Event(enable_timing=True)
            record.timing_end_event = torch.cuda.Event(enable_timing=True)
            record.ready_event = record.timing_end_event
            record.timing_start_event.record(self.copy_stream)
        else:
            record.compute_done_event = None
            record.timing_start_event = None
            record.timing_end_event = None
            record.ready_event = None

        for op in copy_ops:
            if op.kind != GroupCopyKind.D2D:
                continue
            if op.src_buffer_id is None or op.src_slot is None:
                raise RuntimeError("planned D2D operation lost its source")
            self._capture_d2d_integrity_sample(record, op, start=start)
            offset = op.layer_idx - start
            def _copy_d2d() -> None:
                for name in self.WEIGHT_NAMES:
                    self.weight_buffers[name][
                        op.dst_buffer_id, offset, op.dst_slot
                    ].copy_(
                        self.weight_buffers[name][
                            op.src_buffer_id, offset, op.src_slot
                        ],
                        non_blocking=self.copy_stream is not None,
                    )

            if self.copy_stream is None:
                _copy_d2d()
            else:
                with torch.cuda.stream(self.copy_stream):
                    _copy_d2d()

        for offset, slot, prepared in prepared_items:
            info = self._enqueue_prepared_expert(
                prepared=prepared,
                buffer_id=buffer_id,
                offset=offset,
                slot=slot,
                loader=active_loader,
            )
            keepalive = info.get("keepalive")
            if isinstance(keepalive, list):
                record.keepalive.extend(
                    value for value in keepalive if isinstance(value, torch.Tensor)
                )

        if record.timing_end_event is not None:
            record.timing_end_event.record(self.copy_stream)
            if (
                source_buffer_id is not None
                and source_buffer_id != buffer_id
                and record.d2d_experts
            ):
                self.records[source_buffer_id].source_lease_events.append(
                    record.timing_end_event
                )
        record.load_enqueued_at = time.perf_counter()
        if not record.stats_deferred:
            self._account_record_stats(record)
        self._log(
            "group_load_enqueued",
            group_id=group_id,
            buffer_id=buffer_id,
            version=record.version,
            asynchronous=asynchronous,
            layers=[start, end],
            load_started_at=record.load_started_at,
            host_prepare_completed_at=record.host_prepare_completed_at,
            load_enqueued_at=record.load_enqueued_at,
            host_prepare_ms=record.host_prepare_ms,
            prefetch_wait_ms=record.prefetch_wait_ms,
            h2d_bytes=record.h2d_bytes,
            d2d_bytes=record.d2d_bytes,
            retained_bytes=record.retained_bytes,
            h2d_experts=record.h2d_experts,
            d2d_experts=record.d2d_experts,
            retained_experts=record.retained_experts,
            materialization=materialization_kind,
            plan_hash=record.plan_hash,
            ticket_id=record.ticket_id,
            copy_ops=record.copy_ops,
        )
        if not asynchronous or record.ready_event is None:
            self._mark_ready(record, block=True)
        if action_spec is not None:
            record.action_materialization = self._build_action_materialization(
                record,
                source_buffer_id=source_buffer_id,
                materialization=materialization_kind,
                expected_buffer_versions=expected_buffer_versions,
                spec=action_spec,
                physical_source_matches_logical=physical_source_matches_logical,
            )
        elif self._action_group_id is not None:
            self._set_action_materialization(
                record,
                source_buffer_id=source_buffer_id,
                materialization=materialization_kind,
                expected_buffer_versions=expected_buffer_versions,
                physical_source_matches_logical=physical_source_matches_logical,
            )
        return record

    def _mark_ready(self, record: GroupBufferRecord, *, block: bool) -> bool:
        if record.state == GroupBufferState.READY:
            return True
        if record.state != GroupBufferState.LOADING:
            return False
        event = record.ready_event
        if event is None:
            record.state = GroupBufferState.READY
        elif block:
            event.synchronize()
            record.state = GroupBufferState.READY
        elif event.query():
            record.state = GroupBufferState.READY
        else:
            return False
        record.load_completed_at = time.perf_counter()
        record.valid_slots = {
            layer_idx: (True,) * len(experts)
            for layer_idx, experts in record.slot_experts.items()
        }
        self._content_version += 1
        record.content_version = self._content_version
        if record.timing_start_event is not None and record.timing_end_event is not None:
            record.h2d_ms = float(
                record.timing_start_event.elapsed_time(record.timing_end_event)
            )
            if record.stats_accounted:
                self._account_record_timing(record)
        self._log(
            "group_ready",
            group_id=record.group_id,
            buffer_id=record.buffer_id,
            version=record.version,
            load_completed_at=record.load_completed_at,
            host_prepare_ms=record.host_prepare_ms,
            prefetch_wait_ms=record.prefetch_wait_ms,
            h2d_ms=record.h2d_ms,
            h2d_bytes=record.h2d_bytes,
            d2d_bytes=record.d2d_bytes,
            retained_bytes=record.retained_bytes,
            materialization=record.materialization_kind,
            content_version=record.content_version,
        )
        self._verify_record_integrity(record)
        return True

    def _set_layer_mapping(self, layer_idx: int, selected: Sequence[int]) -> None:
        wrapper = self.wrappers[layer_idx]
        selected = tuple(int(x) for x in selected)
        if selected and len(selected) != self.config.slots_per_layer:
            raise ValueError("a committed mapping must fill every physical layer slot")
        mask = torch.zeros(self.config.num_experts, dtype=torch.bool, device="cpu")
        logical = torch.full(
            (self.config.num_experts,), -1, dtype=torch.int32, device="cpu"
        )
        reverse = torch.full(
            (self.config.slots_per_layer,), -1, dtype=torch.int32, device="cpu"
        )
        if selected:
            ids = torch.tensor(selected, dtype=torch.long, device="cpu")
            mask[ids] = True
            logical[ids] = torch.arange(len(selected), dtype=torch.int32)
            reverse[: len(selected)] = ids.to(torch.int32)

        wrapper.gpu_experts_mask.copy_(mask)
        wrapper.logical_to_gpu_index.copy_(logical)
        wrapper.gpu_index_to_logical.copy_(reverse)
        if wrapper.gpu_experts_mask_cuda is not None:
            wrapper.gpu_experts_mask_cuda.copy_(mask, non_blocking=True)
        if wrapper.logical_to_gpu_index_cuda is not None:
            wrapper.logical_to_gpu_index_cuda.copy_(logical, non_blocking=True)
        native = getattr(wrapper, "wrapper", None)
        if native is not None:
            native_mask = getattr(native, "gpu_experts_mask", None)
            if isinstance(native_mask, torch.Tensor):
                native_mask.copy_(mask)
            else:
                native.gpu_experts_mask = mask.clone()
        wrapper._kt_group_cpu_fallback = not bool(selected)

    def _commit_group(self, group_id: int, record: GroupBufferRecord) -> None:
        if record.group_id != group_id or record.state not in {
            GroupBufferState.READY,
            GroupBufferState.REUSABLE,
        }:
            raise RuntimeError(
                f"cannot commit group {group_id} from buffer {record.buffer_id}: "
                f"record_group={record.group_id} state={record.state}"
            )
        if self.active_group_id is not None or self.active_buffer_id is not None:
            raise RuntimeError("previous group must finish before the next atomic commit")
        start, end = self.config.group_range(group_id)
        for layer_idx in range(start, end):
            self._bind_weight_views(layer_idx, record.buffer_id, layer_idx - start)
            self._set_layer_mapping(layer_idx, record.plan[layer_idx])
            self._verify_committed_mapping(layer_idx, record.plan[layer_idx])
        record.state = GroupBufferState.ACTIVE
        record.activated_at = time.perf_counter()
        self.active_group_id = group_id
        self.active_buffer_id = record.buffer_id
        self.active_is_fallback = False
        self.stats["commits"] += 1
        self._log(
            "group_commit",
            group_id=group_id,
            buffer_id=record.buffer_id,
            version=record.version,
            layers=[start, end],
            plan={str(k): list(v) for k, v in record.plan.items()},
            plan_hash=record.plan_hash,
            ticket_id=self._action_ticket_id,
        )

    def _select_action_record(
        self, group_id: int
    ) -> Tuple[Optional[GroupBufferRecord], int, Optional[int]]:
        """Return exact reuse, destination buffer, and delta source buffer."""

        if self._action_group_id is None or self.config.materialization == "full":
            return None, group_id % self.config.buffer_count, None
        for record in self._records_for_group(group_id):
            if self._record_has_exact_plan(record, group_id, self.current_plans):
                return record, record.buffer_id, record.buffer_id
        source = self._latest_record_for_group(group_id)
        if source is None:
            return None, group_id % self.config.buffer_count, None
        return None, 1 - source.buffer_id, source.buffer_id

    def _activate_zero_load(
        self, group_id: int, record: GroupBufferRecord
    ) -> None:
        expected_buffer_versions = tuple(item.version for item in self.records)
        record.ticket_id = self._action_ticket_id
        record.materialization_kind = "zero"
        record.copy_ops = []
        record.host_prepare_ms = 0.0
        record.prefetch_wait_ms = 0.0
        record.h2d_bytes = 0
        record.d2d_bytes = 0
        record.retained_bytes = 0
        record.h2d_experts = 0
        record.d2d_experts = 0
        record.retained_experts = 0
        record.h2d_ms = 0.0
        record.asynchronous = False
        self.stats["zero_loads"] += 1
        self.stats["ready_hits"] += 1
        self._set_action_materialization(
            record,
            source_buffer_id=record.buffer_id,
            materialization="zero",
            expected_buffer_versions=expected_buffer_versions,
        )
        self._log(
            "group_zero_load_reuse",
            group_id=group_id,
            buffer_id=record.buffer_id,
            version=record.version,
            content_version=record.content_version,
            plan_hash=record.plan_hash,
            ticket_id=self._action_ticket_id,
        )

    def _latest_record_for_group_any(
        self, group_id: int
    ) -> Optional[GroupBufferRecord]:
        candidates = [
            record
            for record in self.records
            if record.group_id == int(group_id)
            and record.state
            in {
                GroupBufferState.ACTIVE,
                GroupBufferState.READY,
                GroupBufferState.REUSABLE,
            }
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda record: (
                record.content_version,
                record.version,
                record.state == GroupBufferState.ACTIVE,
            ),
        )

    def _record_has_ready_plan(
        self, record: GroupBufferRecord, spec: GroupActionSpec
    ) -> bool:
        return (
            record.state
            in {
                GroupBufferState.ACTIVE,
                GroupBufferState.READY,
                GroupBufferState.REUSABLE,
            }
            and self._record_has_exact_plan(
                record, spec.group_id, spec.plan_dict
            )
        )

    def _explicit_action_spec(
        self,
        *,
        group_id: int,
        plans: Dict[int, Sequence[int]],
        ticket_id: Optional[int],
        logical_source_plans: Optional[Dict[int, Sequence[int]]],
        logical_source_buffer_id: Optional[int],
        plan_hash: Optional[str],
    ) -> GroupActionSpec:
        start, end = self.config.group_range(int(group_id))
        expected_layers = set(range(start, end))
        if set(plans) != expected_layers:
            raise RuntimeError(
                "lookahead action plan layer mismatch: "
                f"expected={sorted(expected_layers)} got={sorted(plans)}"
            )
        frozen = {
            int(layer_idx): self._normalize_plan(int(layer_idx), plans[layer_idx])
            for layer_idx in range(start, end)
        }
        source_frozen: Dict[int, Tuple[int, ...]] = {}
        if logical_source_plans is not None:
            if set(logical_source_plans) != expected_layers:
                raise RuntimeError("lookahead logical source layer mismatch")
            for layer_idx in range(start, end):
                normalized = self._normalize_plan(
                    layer_idx, logical_source_plans[layer_idx]
                )
                if (
                    tuple(
                        int(value)
                        for value in logical_source_plans[layer_idx]
                    )
                    != normalized
                ):
                    raise RuntimeError(
                        f"lookahead logical source plan is not canonical for layer {layer_idx}"
                    )
                source_frozen[layer_idx] = normalized
        source_record = self._latest_record_for_group_any(group_id)
        self._bound_plans_to_source(
            group_id,
            frozen,
            source_record,
            explicit=True,
            logical_source_plan=(source_frozen or None),
        )
        actual_hash = self._plan_hash(group_id, frozen)
        if plan_hash is not None and str(plan_hash) != actual_hash:
            raise RuntimeError(
                f"lookahead ticket plan hash mismatch: expected={plan_hash} actual={actual_hash}"
            )
        if logical_source_buffer_id is not None and not 0 <= int(
            logical_source_buffer_id
        ) < self.config.buffer_count:
            raise RuntimeError("lookahead logical source buffer ID is out of range")
        return GroupActionSpec(
            group_id=int(group_id),
            plans=self._plans_tuple(frozen),
            ticket_id=None if ticket_id is None else int(ticket_id),
            logical_source_plans=self._plans_tuple(source_frozen),
            logical_source_buffer_id=(
                None
                if logical_source_buffer_id is None
                else int(logical_source_buffer_id)
            ),
            plan_hash=actual_hash,
        )

    def _pending_target_and_source(
        self, spec: GroupActionSpec
    ) -> Tuple[int, Optional[int], bool, Tuple[GroupCopyOp, ...]]:
        active_buffer = self.active_buffer_id
        if active_buffer is None:
            raise RuntimeError("lookahead prefetch requires an active current buffer")
        target_buffer = 1 - int(active_buffer)
        if self.config.materialization == "delta":
            for record in self.records:
                if self._record_has_ready_plan(record, spec):
                    return record.buffer_id, record.buffer_id, True, ()
        source = self._latest_record_for_group_any(spec.group_id)
        source_buffer = (
            None
            if source is None or self.config.materialization == "full"
            else source.buffer_id
        )
        if self.config.materialization == "delta":
            ops = self.plan_materialization(
                spec.group_id,
                target_buffer,
                source_buffer_id=source_buffer,
                plans=spec.plan_dict,
            )
        else:
            start, end = self.config.group_range(spec.group_id)
            ops = tuple(
                GroupCopyOp(
                    kind=GroupCopyKind.H2D,
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                    dst_buffer_id=target_buffer,
                    dst_slot=slot,
                    nbytes=self._expert_nbytes(
                        target_buffer, layer_idx - start, slot
                    ),
                )
                for layer_idx in range(start, end)
                for slot, expert_id in enumerate(spec.plan_dict[layer_idx])
            )
        return target_buffer, source_buffer, False, ops

    def _background_materialize(
        self,
        spec: GroupActionSpec,
        target_buffer_id: int,
        source_buffer_id: Optional[int],
        loader: Any,
    ) -> GroupBufferRecord:
        if self.device is not None and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                return self._enqueue_group_load(
                    spec.group_id,
                    target_buffer_id,
                    asynchronous=True,
                    source_buffer_id=source_buffer_id,
                    action_spec=spec,
                    loader=loader,
                )
        return self._enqueue_group_load(
            spec.group_id,
            target_buffer_id,
            asynchronous=True,
            source_buffer_id=source_buffer_id,
            action_spec=spec,
            loader=loader,
        )

    def prefetch_action(self, payload: Any) -> bool:
        """Freeze a lookahead target and optionally materialize it in background."""

        if not self.step_active or self._action_group_id is None:
            raise RuntimeError("lookahead prefetch requires an active stage action")
        if not isinstance(payload, dict):
            raise TypeError("lookahead action payload must be a dictionary")
        spec = self._explicit_action_spec(
            group_id=int(payload.get("group_id", -1)),
            plans={
                int(layer_idx): tuple(int(value) for value in experts)
                for layer_idx, experts in payload.get("layer_plans", [])
            },
            ticket_id=payload.get("ticket_id"),
            logical_source_plans=(
                {
                    int(layer_idx): tuple(int(value) for value in experts)
                    for layer_idx, experts in payload.get(
                        "logical_source_plans", []
                    )
                }
                if payload.get("logical_source_plans")
                else None
            ),
            logical_source_buffer_id=payload.get("logical_source_buffer_id"),
            plan_hash=payload.get("plan_hash"),
        )
        if spec.ticket_id == self._action_ticket_id:
            return False
        with self._pending_lock:
            if self._pending_action is not None:
                self.stats["pipeline_prefetch_mismatches"] += 1
                raise RuntimeError("a lookahead action is already pending")
        target, source, zero_load, copy_ops = self._pending_target_and_source(spec)
        loader = self.loader
        if loader is None:
            raise RuntimeError("group manager has no expert loader")
        if not zero_load and self.config.load_mode == "async":
            prefetch = getattr(loader, "prefetch_experts", None)
            if callable(prefetch):
                by_layer: Dict[int, List[int]] = {}
                for op in copy_ops:
                    if op.kind == GroupCopyKind.H2D:
                        by_layer.setdefault(op.layer_idx, []).append(op.expert_id)
                submitted = sum(
                    int(prefetch(layer_idx, experts) or 0)
                    for layer_idx, experts in by_layer.items()
                )
                self.stats["host_prefetch_experts_submitted"] += submitted
        if zero_load:
            future: Future = Future()
            record = self.records[target]
            future.set_result(record)
        elif self.config.load_mode == "async":
            if self._pipeline_executor is None:
                self._pipeline_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="kt-group-pipeline"
                )
            future = self._pipeline_executor.submit(
                self._background_materialize,
                spec,
                target,
                source,
                loader,
            )
        else:
            future = Future()
            future.set_result(None)
        pending = PendingGroupAction(
            spec=spec,
            target_buffer_id=target,
            future=future,
            submitted_at=time.perf_counter(),
            zero_load=zero_load,
            source_buffer_id=source,
            loader=loader,
            deferred_sync=(
                self.config.load_mode == "sync" and not zero_load
            ),
        )
        with self._pending_lock:
            self._pending_action = pending
        if self.config.load_mode == "async":
            self.stats["pipeline_prefetch_submitted"] += 1
        else:
            self.stats["pipeline_sync_hints"] += 1
        if zero_load and self.config.load_mode == "async":
            self.stats["pipeline_prefetch_zero_loads"] += 1
        self._log(
            (
                "pipeline_prefetch_submitted"
                if self.config.load_mode == "async"
                else "pipeline_sync_hint_frozen"
            ),
            ticket_id=spec.ticket_id,
            group_id=spec.group_id,
            target_buffer_id=target,
            source_buffer_id=source,
            zero_load=zero_load,
            copy_ops=[op.to_dict() for op in copy_ops],
        )
        return True

    def _take_pending_action(self) -> Optional[PendingGroupAction]:
        with self._pending_lock:
            pending = self._pending_action
            self._pending_action = None
        return pending

    def _discard_pending_action(
        self, pending: PendingGroupAction, *, reason: str
    ) -> None:
        record: Optional[GroupBufferRecord] = None
        try:
            value = pending.future.result()
            if isinstance(value, GroupBufferRecord):
                record = value
        except Exception as exc:
            self.stats["pipeline_prefetch_failures"] += 1
            self._log(
                "pipeline_prefetch_failed",
                ticket_id=pending.spec.ticket_id,
                group_id=pending.spec.group_id,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        if record is None and pending.target_buffer_id is not None:
            candidate = self.records[pending.target_buffer_id]
            if (
                candidate.group_id == pending.spec.group_id
                and candidate.ticket_id == pending.spec.ticket_id
                and candidate.state
                in {
                    GroupBufferState.LOADING,
                    GroupBufferState.READY,
                    GroupBufferState.REUSABLE,
                }
            ):
                record = candidate
        if record is not None and record.state != GroupBufferState.ACTIVE:
            self._invalidate_record(record)
        self._log(
            "pipeline_prefetch_discarded",
            ticket_id=pending.spec.ticket_id,
            group_id=pending.spec.group_id,
            reason=reason,
        )

    def _adopt_pending_action(self, spec: GroupActionSpec) -> bool:
        pending = self._take_pending_action()
        if pending is None:
            return False
        if not self._action_specs_match(pending.spec, spec):
            self.stats["pipeline_prefetch_mismatches"] += 1
            self._discard_pending_action(pending, reason="ticket_mismatch")
            return False
        t0 = time.perf_counter()
        try:
            value = pending.future.result()
        except Exception as exc:
            self._discard_pending_action(pending, reason="future_error")
            self._log(
                "pipeline_prefetch_fallback",
                ticket_id=spec.ticket_id,
                group_id=spec.group_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False
        if pending.deferred_sync:
            if pending.target_buffer_id is None or pending.loader is None:
                raise RuntimeError("sync lookahead lost its target or loader")
            value = self._enqueue_group_load(
                spec.group_id,
                pending.target_buffer_id,
                asynchronous=False,
                source_buffer_id=pending.source_buffer_id,
                loader=pending.loader,
            )
        elif self.config.load_mode == "async":
            self.stats["pipeline_enqueue_tail_ms"] += (
                time.perf_counter() - t0
            ) * 1000.0
        if pending.zero_load:
            record = value
            if not isinstance(record, GroupBufferRecord) or not self._record_has_ready_plan(
                record, spec
            ):
                self.stats["pipeline_prefetch_mismatches"] += 1
                self._discard_pending_action(pending, reason="zero_plan_changed")
                return False
            self._adopted_prefetch_record = record
            self._adopted_prefetch_zero_load = True
        else:
            record = value
            if not isinstance(record, GroupBufferRecord) or (
                record.group_id != spec.group_id
                or record.plan_hash != spec.plan_hash
                or record.ticket_id != spec.ticket_id
            ):
                self.stats["pipeline_prefetch_mismatches"] += 1
                self._discard_pending_action(pending, reason="record_mismatch")
                return False
            self._account_record_stats(record)
            self._adopted_prefetch_record = record
            self._adopted_prefetch_zero_load = False
            self._action_materialization = dict(
                record.action_materialization or {}
            )
        if self.config.load_mode == "async":
            self.stats["pipeline_prefetch_adopted"] += 1
        self._log(
            (
                "pipeline_sync_hint_adopted"
                if self.config.load_mode == "sync"
                else "pipeline_prefetch_adopted"
            ),
            ticket_id=spec.ticket_id,
            group_id=spec.group_id,
            buffer_id=record.buffer_id,
            zero_load=pending.zero_load,
            enqueue_tail_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return True

    def submit_host_prefetch_for_next_group(self, current_group_id: int) -> None:
        """Submit CPU tensor reads while the current group is executing.

        This only starts host-side work.  The physical destination buffer is
        untouched until ``finish_group`` confirms the active buffer is free.
        """

        if self.config.load_mode != "async":
            return
        next_group_id = current_group_id + 1
        if next_group_id >= self.config.num_groups or self.loader is None:
            return
        prefetch = getattr(self.loader, "prefetch_experts", None)
        if not callable(prefetch):
            return
        start, end = self.config.group_range(next_group_id)
        submitted = 0
        for layer_idx in range(start, end):
            submitted += int(prefetch(
                layer_idx, list(self.current_plans[layer_idx])
            ) or 0)
        self.stats["host_prefetch_experts_submitted"] += submitted
        self._log(
            "group_host_prefetch_submitted",
            current_group_id=current_group_id,
            next_group_id=next_group_id,
            layers=[start, end],
            experts_submitted=submitted,
        )

    def enqueue_next_group_h2d_after_host_prefetch(self, current_group_id: int) -> None:
        """Enqueue next-group DMA while the active group remains executable.

        With exactly two buffers, group ``g+1`` always occupies the inactive
        buffer while ``g`` is active.  This is the only point where both host
        preparation and H2D can overlap the current group without risking an
        overwrite.  If host work is still incomplete, ``_enqueue_group_load``
        may wait only for that host work; the resulting boundary condition is
        still represented by the record event.
        """

        if self.config.load_mode != "async":
            return
        next_group_id = current_group_id + 1
        if next_group_id >= self.config.num_groups:
            return
        buffer_id = next_group_id % self.config.buffer_count
        if buffer_id == self.active_buffer_id:
            raise RuntimeError(
                "next-group H2D selected the ACTIVE buffer; double-buffer "
                "invariant is broken"
            )
        record = self.records[buffer_id]
        if record.group_id == next_group_id and record.state in {
            GroupBufferState.LOADING,
            GroupBufferState.READY,
        }:
            return
        if record.state == GroupBufferState.LOADING and not self._mark_ready(
            record, block=False
        ):
            self.stats["prefetch_deferred_buffer_busy"] += 1
            self._log(
                "group_prefetch_deferred_buffer_busy",
                current_group_id=current_group_id,
                next_group_id=next_group_id,
                buffer_id=buffer_id,
                loading_group_id=record.group_id,
            )
            return
        self._enqueue_group_load(next_group_id, buffer_id, asynchronous=True)

    def _activate_cpu_fallback(self, group_id: int) -> None:
        if self.active_group_id is not None or self.active_buffer_id is not None:
            raise RuntimeError("previous group must finish before CPU fallback")
        start, end = self.config.group_range(group_id)
        for layer_idx in range(start, end):
            self._set_layer_mapping(layer_idx, ())
        self.active_group_id = group_id
        self.active_buffer_id = None
        self.active_is_fallback = True
        self.stats["cpu_fallback_groups"] += 1
        self._log("group_cpu_fallback", group_id=group_id, layers=[start, end])

    def _schedule_next_group(self, current_group_id: int) -> None:
        if self.config.load_mode != "async":
            return
        next_group_id = current_group_id + 1
        if next_group_id >= self.config.num_groups:
            return
        buffer_id = next_group_id % self.config.buffer_count
        record = self.records[buffer_id]
        if record.group_id == next_group_id and record.state in {
            GroupBufferState.LOADING,
            GroupBufferState.READY,
        }:
            return
        if record.state == GroupBufferState.LOADING and not self._mark_ready(
            record, block=False
        ):
            self.stats["prefetch_deferred_buffer_busy"] += 1
            self._log(
                "group_prefetch_deferred_buffer_busy",
                current_group_id=current_group_id,
                next_group_id=next_group_id,
                buffer_id=buffer_id,
                loading_group_id=record.group_id,
            )
            return
        self._enqueue_group_load(next_group_id, buffer_id, asynchronous=True)

    def activate_group(self, group_id: int) -> str:
        if not self.step_active:
            raise RuntimeError("activate_group requires begin_step")
        expected = self._last_finished_group + 1
        if group_id != expected:
            raise RuntimeError(
                f"groups must execute in order: expected {expected}, got {group_id}"
            )
        if self.active_group_id is not None:
            raise RuntimeError("cannot activate a group before finishing the current group")

        adopted_record = self._adopted_prefetch_record
        adopted_zero_load = self._adopted_prefetch_zero_load
        self._adopted_prefetch_record = None
        self._adopted_prefetch_zero_load = False
        if adopted_record is not None:
            exact_record = adopted_record if adopted_zero_load else None
            buffer_id = adopted_record.buffer_id
            source_buffer_id = None
            if adopted_zero_load:
                self._activate_zero_load(group_id, adopted_record)
                self._commit_group(group_id, adopted_record)
                return "gpu"
        else:
            exact_record, buffer_id, source_buffer_id = self._select_action_record(group_id)
        if exact_record is not None:
            self._activate_zero_load(group_id, exact_record)
            self._commit_group(group_id, exact_record)
            return "gpu"
        record = self.records[buffer_id]
        if record.group_id != group_id or record.state not in {
            GroupBufferState.LOADING,
            GroupBufferState.READY,
        }:
            if record.state == GroupBufferState.LOADING and not self._mark_ready(
                record, block=False
            ):
                self.stats["ready_misses"] += 1
                if self.config.miss_policy == "cpu_fallback":
                    self._activate_cpu_fallback(group_id)
                    return "cpu_fallback"
            record = self._enqueue_group_load(
                group_id,
                buffer_id,
                asynchronous=False,
                source_buffer_id=source_buffer_id,
            )
        elif record.state == GroupBufferState.LOADING:
            ready = self._mark_ready(record, block=False)
            if ready:
                self.stats["ready_hits"] += 1
                if record.asynchronous and record.h2d_ms is not None:
                    self.stats["overlap_ms"] += record.h2d_ms
            else:
                self.stats["ready_misses"] += 1
                if self.config.miss_policy == "block":
                    t0 = time.perf_counter()
                    self._mark_ready(record, block=True)
                    block_ms = (time.perf_counter() - t0) * 1000.0
                    self.stats["block_count"] += 1
                    self.stats["block_ms"] += block_ms
                    self.stats["uncovered_boundary_tail_ms"] += block_ms
                    if record.asynchronous and record.h2d_ms is not None:
                        self.stats["overlap_ms"] += max(
                            0.0, record.h2d_ms - block_ms
                        )
                    self._log(
                        "group_block_wait",
                        group_id=group_id,
                        buffer_id=buffer_id,
                        block_ms=block_ms,
                    )
                else:
                    self._activate_cpu_fallback(group_id)
                    self._schedule_next_group(group_id)
                    return "cpu_fallback"
        else:
            self.stats["ready_hits"] += 1
            if record.asynchronous and record.h2d_ms is not None:
                self.stats["overlap_ms"] += record.h2d_ms

        self._commit_group(group_id, record)
        return "gpu"

    def finish_group(self, group_id: int) -> None:
        if self.active_group_id != group_id:
            raise RuntimeError(
                f"finish_group mismatch: active={self.active_group_id}, got={group_id}"
            )
        if self.active_is_fallback:
            buffer_id = None
            compute_window_ms = None
        else:
            if self.active_buffer_id is None:
                raise RuntimeError("ACTIVE group has no buffer")
            record = self.records[self.active_buffer_id]
            if record.state != GroupBufferState.ACTIVE or record.group_id != group_id:
                raise RuntimeError("active buffer record is inconsistent")
            buffer_id = record.buffer_id
            compute_window_ms = (
                (time.perf_counter() - record.activated_at) * 1000.0
                if record.activated_at is not None
                else None
            )
            record.state = GroupBufferState.REUSABLE
            self._last_finished_buffer_id = record.buffer_id
            if self.device is not None and self.device.type == "cuda":
                record.compute_done_event = torch.cuda.Event()
                record.compute_done_event.record(
                    torch.cuda.current_stream(device=self.device)
                )
        self.stats["groups_executed"] += 1
        self._last_finished_group = group_id
        self._log(
            "group_finish",
            group_id=group_id,
            buffer_id=buffer_id,
            cpu_fallback=self.active_is_fallback,
            compute_window_ms=compute_window_ms,
        )
        self.active_group_id = None
        self.active_buffer_id = None
        self.active_is_fallback = False
        # Stage-ready actions submit an explicit arbitrary-ticket lookahead from
        # the worker.  The legacy step path still uses the fixed g -> g+1
        # prefetch below; scheduling it for a ticket action would overwrite the
        # arbitrary lookahead contract.
        if self._action_group_id is None:
            self._schedule_next_group(group_id)

    def assert_layer_ready_for_apply(
        self, layer_idx: int, layer: torch.nn.Module
    ) -> None:
        expected_group = int(layer_idx) // self.config.group_size
        if self.active_group_id != expected_group:
            raise RuntimeError(
                f"layer {layer_idx} belongs to group {expected_group}, but active "
                f"group is {self.active_group_id}"
            )
        if self.layers.get(int(layer_idx)) is not layer:
            raise RuntimeError(f"layer {layer_idx} is not the registered runtime layer")
        if self.active_is_fallback:
            wrapper = self.wrappers[int(layer_idx)]
            if bool(wrapper.gpu_experts_mask.any().item()):
                raise RuntimeError(
                    f"CPU fallback layer {layer_idx} still exposes GPU experts"
                )
            return
        if self.active_buffer_id is None:
            raise RuntimeError("non-fallback group has no active buffer")
        start, _ = self.config.group_range(expected_group)
        offset = int(layer_idx) - start
        for name in self.WEIGHT_NAMES:
            param = getattr(layer, name)
            expected = self.weight_buffers[name][self.active_buffer_id, offset]
            if param.data_ptr() != expected.data_ptr():
                raise RuntimeError(
                    f"stale {name} binding for layer {layer_idx}: "
                    f"actual={param.data_ptr()} expected={expected.data_ptr()}"
                )

    def metrics_snapshot(self) -> Dict[str, Any]:
        metrics = {
            "step": self.step_id,
            "physical_slots": self.physical_slots,
            "allocated_bytes": self.allocated_bytes,
            "group_size": self.config.group_size,
            "slots_per_layer": self.config.slots_per_layer,
            "buffer_count": self.config.buffer_count,
            "load_mode": self.config.load_mode,
            "miss_policy": self.config.miss_policy,
            "materialization": self.config.materialization,
            "max_replacements": self.config.max_replacements,
            **{
                key: int(value) if float(value).is_integer() else float(value)
                for key, value in self.stats.items()
            },
        }
        if self.loader is not None:
            for name in (
                "prefetch_submitted",
                "prefetch_cache_hits",
                "prefetch_future_hits",
                "prefetch_direct_loads",
                "prefetch_wait_ms_total",
                "cpu_tensor_cache_hits",
                "cpu_tensor_cache_misses",
                "cpu_tensor_host_register_successes",
                "cpu_tensor_host_register_failures",
            ):
                value = getattr(self.loader, name, None)
                if value is None:
                    continue
                metrics[f"loader_{name}"] = (
                    int(value) if isinstance(value, int) else float(value)
                )
        return metrics

    def end_step(self) -> Dict[str, Any]:
        if not self.step_active:
            raise RuntimeError("no group step is active")
        if self.active_group_id is not None:
            raise RuntimeError("cannot end a step while a group is active")
        if self._last_finished_group != self.config.num_groups - 1:
            raise RuntimeError(
                f"step ended after group {self._last_finished_group}, expected "
                f"{self.config.num_groups - 1}"
            )
        pending = self._take_pending_action()
        if pending is not None:
            self._discard_pending_action(pending, reason="step_end")
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        self._drain_inflight_keepalive(block=True)
        for record in self.records:
            if record.state == GroupBufferState.LOADING:
                self._mark_ready(record, block=True)
            record.keepalive.clear()
            record.integrity_samples.clear()
            if record.state == GroupBufferState.ACTIVE:
                raise RuntimeError("ACTIVE buffer survived step completion")
        self.step_active = False
        self._action_group_id = None
        self._action_ticket_id = None
        self._action_plan_hash = None
        self._action_materialization = None
        self._action_logical_source_buffer_id = None
        self._action_logical_source_plan = None
        self._adopted_prefetch_record = None
        self._adopted_prefetch_zero_load = False
        self.current_plans = {}
        self.loader = None
        self.provider = None
        metrics = self.metrics_snapshot()
        self._log("step_end", metrics=metrics)
        return metrics

    def end_action(self, group_id: int) -> Dict[str, Any]:
        group_id = int(group_id)
        if not self.step_active or self._action_group_id != group_id:
            raise RuntimeError(
                "no matching group action is active: "
                f"active={self._action_group_id} requested={group_id}"
            )
        if self.active_group_id is not None:
            raise RuntimeError("cannot end an action while its group is active")
        if self._last_finished_group != group_id:
            raise RuntimeError(
                f"action ended after group {self._last_finished_group}, expected {group_id}"
            )
        # The lookahead copy stream is deliberately left running.  Only the
        # foreground action's record is retired here; the pending record owns
        # its host tensors and CUDA event until the next action adopts it.
        self._drain_inflight_keepalive(block=False)
        for record in self.records:
            if record.state == GroupBufferState.ACTIVE:
                raise RuntimeError("ACTIVE buffer survived action completion")
            if (
                record.group_id == group_id
                and record.state == GroupBufferState.REUSABLE
                and record.ticket_id == self._action_ticket_id
            ):
                record.keepalive.clear()
                record.integrity_samples.clear()
        if self.config.load_mode == "async":
            self.stats["pipeline_end_action_nonblocking"] += 1
        metrics = self.metrics_snapshot()
        metrics["action_materialization"] = dict(
            self._action_materialization or {}
        )
        self.step_active = False
        self._action_group_id = None
        self._action_ticket_id = None
        self._action_plan_hash = None
        self._action_materialization = None
        self._action_logical_source_buffer_id = None
        self._action_logical_source_plan = None
        self._adopted_prefetch_record = None
        self._adopted_prefetch_zero_load = False
        self.current_plans = {}
        self.loader = None
        self.provider = None
        self._log("action_end", action_group_id=group_id, metrics=metrics)
        return metrics

    def abort_step(self, reason: str) -> None:
        pending = self._take_pending_action()
        if pending is not None:
            self._discard_pending_action(pending, reason="abort")
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        self._drain_inflight_keepalive(block=True)
        for record in self.records:
            if record.state in {
                GroupBufferState.LOADING,
                GroupBufferState.READY,
                GroupBufferState.ACTIVE,
            }:
                record.state = GroupBufferState.REUSABLE
            record.keepalive.clear()
            record.integrity_samples.clear()
        self._log(
            "step_abort",
            reason=str(reason),
            active_group=self.active_group_id,
            active_buffer=self.active_buffer_id,
        )
        self.step_active = False
        self._action_group_id = None
        self._action_ticket_id = None
        self._action_plan_hash = None
        self._action_materialization = None
        self._action_logical_source_buffer_id = None
        self._action_logical_source_plan = None
        self._adopted_prefetch_record = None
        self._adopted_prefetch_zero_load = False
        self.active_group_id = None
        self.active_buffer_id = None
        self.active_is_fallback = False
        self.current_plans = {}
        self.loader = None
        self.provider = None


__all__ = [
    "GroupCopyKind",
    "GroupCopyOp",
    "GroupBufferRecord",
    "GroupBufferState",
    "KTGroupBufferConfig",
    "KTGroupExpertBufferManager",
]
