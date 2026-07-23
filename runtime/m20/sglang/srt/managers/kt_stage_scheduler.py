# SPDX-License-Identifier: Apache-2.0
"""Deterministic stage-ready scheduling primitives for M20-B.

This module deliberately does not import the SGLang scheduler or torch.  It
owns dependency/fairness state and immutable action tickets; tensor ownership
stays in the caller-provided ``payload`` of :class:`ChunkStageState`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


ACTION_TRACE_SCHEMA_VERSION = 1


class StageState(str, Enum):
    NEW = "NEW"
    READY = "READY"
    RESERVED = "RESERVED"
    RUNNING = "RUNNING"
    FINALIZING = "FINALIZING"
    DONE = "DONE"
    FAILED = "FAILED"


class CopyKind(str, Enum):
    RETAIN = "retain"
    D2D = "d2d"
    H2D = "h2d"


@dataclass(frozen=True)
class CopyOp:
    kind: CopyKind
    layer_idx: int
    expert_id: int
    dst_buffer_id: int
    dst_slot: int
    nbytes: int
    src_buffer_id: Optional[int] = None
    src_slot: Optional[int] = None

    def __post_init__(self) -> None:
        if self.layer_idx < 0 or self.expert_id < 0:
            raise ValueError("copy op layer/expert IDs must be non-negative")
        if self.dst_buffer_id < 0 or self.dst_slot < 0 or self.nbytes < 0:
            raise ValueError("copy op destination and nbytes must be non-negative")
        if self.kind == CopyKind.D2D and (
            self.src_buffer_id is None or self.src_slot is None
        ):
            raise ValueError("D2D copy op requires a source buffer and slot")

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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CopyOp":
        return cls(
            kind=CopyKind(str(value["kind"])),
            layer_idx=int(value["layer_idx"]),
            expert_id=int(value["expert_id"]),
            dst_buffer_id=int(value["dst_buffer_id"]),
            dst_slot=int(value["dst_slot"]),
            nbytes=int(value["nbytes"]),
            src_buffer_id=(
                None
                if value.get("src_buffer_id") is None
                else int(value["src_buffer_id"])
            ),
            src_slot=(
                None if value.get("src_slot") is None else int(value["src_slot"])
            ),
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    compute_gain_ms: float = 0.0
    materialization_ms: float = 0.0
    copy_contention_ms: float = 0.0
    eviction_loss_ms: float = 0.0
    queue_penalty_ms: float = 0.0
    covered_route_entries: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not math.isfinite(value):
                raise ValueError(f"score component {name} must be finite")
        if self.covered_route_entries < 0:
            raise ValueError("covered route entries must be non-negative")

    @property
    def net_gain_ms(self) -> float:
        return (
            self.compute_gain_ms
            - self.materialization_ms
            - self.copy_contention_ms
            - self.eviction_loss_ms
            - self.queue_penalty_ms
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "compute_gain_ms": float(self.compute_gain_ms),
            "materialization_ms": float(self.materialization_ms),
            "copy_contention_ms": float(self.copy_contention_ms),
            "eviction_loss_ms": float(self.eviction_loss_ms),
            "queue_penalty_ms": float(self.queue_penalty_ms),
            "covered_route_entries": float(self.covered_route_entries),
            "net_gain_ms": float(self.net_gain_ms),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreBreakdown":
        return cls(
            compute_gain_ms=float(value.get("compute_gain_ms", 0.0)),
            materialization_ms=float(value.get("materialization_ms", 0.0)),
            copy_contention_ms=float(value.get("copy_contention_ms", 0.0)),
            eviction_loss_ms=float(value.get("eviction_loss_ms", 0.0)),
            queue_penalty_ms=float(value.get("queue_penalty_ms", 0.0)),
            covered_route_entries=float(
                value.get("covered_route_entries", 0.0)
            ),
        )


LayerPlans = Tuple[Tuple[int, Tuple[int, ...]], ...]


def normalize_layer_plans(
    plans: Mapping[int, Sequence[int]] | Iterable[Tuple[int, Sequence[int]]]
) -> LayerPlans:
    items = plans.items() if isinstance(plans, Mapping) else plans
    normalized: List[Tuple[int, Tuple[int, ...]]] = []
    seen_layers = set()
    for raw_layer, raw_experts in items:
        layer_idx = int(raw_layer)
        experts = tuple(int(value) for value in raw_experts)
        if layer_idx < 0 or layer_idx in seen_layers:
            raise ValueError(f"invalid or duplicate layer plan: {layer_idx}")
        if len(experts) != len(set(experts)) or any(value < 0 for value in experts):
            raise ValueError(f"layer {layer_idx} plan must contain unique non-negative IDs")
        seen_layers.add(layer_idx)
        normalized.append((layer_idx, experts))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def compute_plan_hash(group_id: int, layer_plans: LayerPlans) -> str:
    return sha256_json(
        {
            "group_id": int(group_id),
            "layer_plans": [
                [layer_idx, list(experts)] for layer_idx, experts in layer_plans
            ],
        }
    )


@dataclass(frozen=True)
class NextActionTicket:
    ticket_id: int
    queue_epoch: int
    group_id: int
    state_ids: Tuple[int, ...]
    state_versions: Tuple[int, ...]
    request_ids: Tuple[str, ...]
    chunk_indices: Tuple[int, ...]
    token_spans: Tuple[Tuple[int, int], ...]
    token_count: int
    layer_plans: LayerPlans
    plan_hash: str
    policy: str
    provider_version: str
    confidence: float
    logical_source_plans: LayerPlans = ()
    logical_source_buffer_id: Optional[int] = None
    target_buffer_id: Optional[int] = None
    expected_buffer_versions: Tuple[int, int] = (-1, -1)
    copy_ops: Tuple[CopyOp, ...] = ()
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    fallback: str = ""

    def __post_init__(self) -> None:
        size = len(self.state_ids)
        if self.ticket_id < 0 or self.queue_epoch < 0 or self.group_id < 0:
            raise ValueError("ticket, queue epoch, and group IDs must be non-negative")
        if size <= 0:
            raise ValueError("an action ticket must contain at least one state")
        if len(set(self.state_ids)) != size:
            raise ValueError("an action ticket cannot contain duplicate states")
        for values in (
            self.state_versions,
            self.request_ids,
            self.chunk_indices,
            self.token_spans,
        ):
            if len(values) != size:
                raise ValueError("ticket state metadata lengths do not match")
        if len(set(self.request_ids)) != size:
            raise ValueError("M20-B v1 permits only one chunk per request in a ticket")
        if any(start < 0 or end <= start for start, end in self.token_spans):
            raise ValueError("ticket token spans must be non-empty and non-negative")
        expected_tokens = sum(end - start for start, end in self.token_spans)
        if self.token_count != expected_tokens:
            raise ValueError(
                f"ticket token count mismatch: {self.token_count} != {expected_tokens}"
            )
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("ticket confidence must be finite and in [0, 1]")
        expected_hash = compute_plan_hash(self.group_id, self.layer_plans)
        if self.plan_hash != expected_hash:
            raise ValueError("ticket plan hash does not match its group/layer plans")
        if self.logical_source_plans and {
            layer_idx for layer_idx, _ in self.logical_source_plans
        } != {layer_idx for layer_idx, _ in self.layer_plans}:
            raise ValueError("ticket logical source and target layers must match")
        if (
            self.logical_source_buffer_id is not None
            and self.logical_source_buffer_id < 0
        ):
            raise ValueError("logical source buffer ID must be non-negative")
        if self.target_buffer_id is not None and self.target_buffer_id < 0:
            raise ValueError("target buffer ID must be non-negative")
        if len(self.expected_buffer_versions) != 2:
            raise ValueError("M20-B ticket must carry exactly two buffer versions")

    @classmethod
    def create(
        cls,
        *,
        ticket_id: int,
        queue_epoch: int,
        group_id: int,
        states: Sequence["ChunkStageState"],
        state_versions: Sequence[int],
        layer_plans: Mapping[int, Sequence[int]] | Iterable[Tuple[int, Sequence[int]]],
        policy: str,
        provider_version: str = "",
        confidence: float = 1.0,
        target_buffer_id: Optional[int] = None,
        logical_source_plans: Mapping[int, Sequence[int]]
        | Iterable[Tuple[int, Sequence[int]]] = (),
        logical_source_buffer_id: Optional[int] = None,
        expected_buffer_versions: Tuple[int, int] = (-1, -1),
        copy_ops: Sequence[CopyOp] = (),
        score: Optional[ScoreBreakdown] = None,
        fallback: str = "",
    ) -> "NextActionTicket":
        normalized_plans = normalize_layer_plans(layer_plans)
        normalized_source_plans = normalize_layer_plans(logical_source_plans)
        return cls(
            ticket_id=int(ticket_id),
            queue_epoch=int(queue_epoch),
            group_id=int(group_id),
            state_ids=tuple(state.state_id for state in states),
            state_versions=tuple(int(value) for value in state_versions),
            request_ids=tuple(state.request_id for state in states),
            chunk_indices=tuple(state.chunk_index for state in states),
            token_spans=tuple(
                (state.token_start, state.token_end) for state in states
            ),
            token_count=sum(state.token_end - state.token_start for state in states),
            layer_plans=normalized_plans,
            plan_hash=compute_plan_hash(group_id, normalized_plans),
            policy=str(policy),
            provider_version=str(provider_version),
            confidence=float(confidence),
            logical_source_plans=normalized_source_plans,
            logical_source_buffer_id=(
                None
                if logical_source_buffer_id is None
                else int(logical_source_buffer_id)
            ),
            target_buffer_id=target_buffer_id,
            expected_buffer_versions=tuple(
                int(value) for value in expected_buffer_versions
            ),
            copy_ops=tuple(copy_ops),
            score=score if score is not None else ScoreBreakdown(),
            fallback=str(fallback),
        )

    @property
    def plan_dict(self) -> Dict[int, Tuple[int, ...]]:
        return dict(self.layer_plans)

    def materialize(
        self,
        layer_plans: Mapping[int, Sequence[int]]
        | Iterable[Tuple[int, Sequence[int]]],
        *,
        plan_hash: Optional[str] = None,
        target_buffer_id: Optional[int] = None,
        expected_buffer_versions: Optional[Sequence[int]] = None,
        copy_ops: Optional[Sequence[CopyOp]] = None,
    ) -> "NextActionTicket":
        """Bind the immutable ticket to the placement frozen by the worker."""

        normalized = normalize_layer_plans(layer_plans)
        if not normalized:
            raise ValueError("a materialized action ticket must contain layer plans")
        materialized_hash = compute_plan_hash(self.group_id, normalized)
        if plan_hash is not None and str(plan_hash) != materialized_hash:
            raise ValueError("worker plan hash does not match its materialized plans")
        if self.layer_plans and (
            self.layer_plans != normalized or self.plan_hash != materialized_hash
        ):
            raise ValueError(
                f"ticket {self.ticket_id} worker placement diverged from replay plans"
            )
        normalized_copy_ops = self.copy_ops if copy_ops is None else tuple(copy_ops)
        normalized_target = (
            self.target_buffer_id
            if target_buffer_id is None
            else int(target_buffer_id)
        )
        normalized_versions = (
            self.expected_buffer_versions
            if expected_buffer_versions is None
            else tuple(int(value) for value in expected_buffer_versions)
        )
        # Physical materialization is deliberately re-derived during replay:
        # a full planner trace can be executed by a delta runtime while the
        # immutable action order and logical layer plans remain identical.
        if self.layer_plans and copy_ops is not None:
            normalized_copy_ops = tuple(copy_ops)
        return replace(
            self,
            layer_plans=normalized,
            plan_hash=materialized_hash,
            target_buffer_id=normalized_target,
            expected_buffer_versions=normalized_versions,
            copy_ops=normalized_copy_ops,
        )

    def execution_payload(self) -> Dict[str, Any]:
        """Small JSON-compatible contract carried through ForwardBatch metadata."""

        return {
            "ticket_id": self.ticket_id,
            "group_id": self.group_id,
            "layer_plans": [
                [layer_idx, list(experts)] for layer_idx, experts in self.layer_plans
            ],
            "plan_hash": self.plan_hash,
            "logical_source_plans": [
                [layer_idx, list(experts)]
                for layer_idx, experts in self.logical_source_plans
            ],
            "logical_source_buffer_id": self.logical_source_buffer_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "queue_epoch": self.queue_epoch,
            "group_id": self.group_id,
            "state_ids": list(self.state_ids),
            "state_versions": list(self.state_versions),
            "request_ids": list(self.request_ids),
            "chunk_indices": list(self.chunk_indices),
            "token_spans": [list(value) for value in self.token_spans],
            "token_count": self.token_count,
            "layer_plans": [
                [layer_idx, list(experts)] for layer_idx, experts in self.layer_plans
            ],
            "plan_hash": self.plan_hash,
            "policy": self.policy,
            "provider_version": self.provider_version,
            "confidence": self.confidence,
            "logical_source_plans": [
                [layer_idx, list(experts)]
                for layer_idx, experts in self.logical_source_plans
            ],
            "logical_source_buffer_id": self.logical_source_buffer_id,
            "target_buffer_id": self.target_buffer_id,
            "expected_buffer_versions": list(self.expected_buffer_versions),
            "copy_ops": [op.to_dict() for op in self.copy_ops],
            "score": self.score.to_dict(),
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextActionTicket":
        layer_plans = normalize_layer_plans(
            (int(layer_idx), experts)
            for layer_idx, experts in value.get("layer_plans", [])
        )
        logical_source_plans = normalize_layer_plans(
            (int(layer_idx), experts)
            for layer_idx, experts in value.get("logical_source_plans", [])
        )
        return cls(
            ticket_id=int(value["ticket_id"]),
            queue_epoch=int(value["queue_epoch"]),
            group_id=int(value["group_id"]),
            state_ids=tuple(int(item) for item in value["state_ids"]),
            state_versions=tuple(int(item) for item in value["state_versions"]),
            request_ids=tuple(str(item) for item in value["request_ids"]),
            chunk_indices=tuple(int(item) for item in value["chunk_indices"]),
            token_spans=tuple(
                (int(item[0]), int(item[1])) for item in value["token_spans"]
            ),
            token_count=int(value["token_count"]),
            layer_plans=layer_plans,
            plan_hash=str(value["plan_hash"]),
            policy=str(value["policy"]),
            provider_version=str(value.get("provider_version", "")),
            confidence=float(value.get("confidence", 1.0)),
            logical_source_plans=logical_source_plans,
            logical_source_buffer_id=(
                None
                if value.get("logical_source_buffer_id") is None
                else int(value["logical_source_buffer_id"])
            ),
            target_buffer_id=(
                None
                if value.get("target_buffer_id") is None
                else int(value["target_buffer_id"])
            ),
            expected_buffer_versions=tuple(
                int(item) for item in value.get("expected_buffer_versions", (-1, -1))
            ),
            copy_ops=tuple(
                CopyOp.from_dict(item) for item in value.get("copy_ops", [])
            ),
            score=ScoreBreakdown.from_dict(value.get("score", {})),
            fallback=str(value.get("fallback", "")),
        )


@dataclass
class ChunkStageState:
    state_id: int
    request_id: str
    chunk_index: int
    token_start: int
    token_end: int
    group_id: int
    num_groups: int
    enqueue_seq: int
    ready_since: float
    deadline_at: float
    payload: Any = None
    demand_by_layer: Optional[Mapping[int, Mapping[int, float]]] = None
    confidence_by_layer: Optional[Mapping[int, float]] = None
    state_version: int = 0
    status: StageState = StageState.NEW
    reserved_ticket_id: Optional[int] = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.state_id < 0 or self.chunk_index < 0 or self.enqueue_seq < 0:
            raise ValueError("state/chunk/enqueue IDs must be non-negative")
        if not self.request_id:
            raise ValueError("request ID must be non-empty")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("chunk token span must be non-empty and non-negative")
        if self.num_groups <= 0 or not 0 <= self.group_id < self.num_groups:
            raise ValueError("chunk group ID is outside its configured range")
        if self.deadline_at < self.ready_since:
            raise ValueError("chunk deadline cannot precede ready time")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class StageSchedulerConfig:
    num_groups: int
    cohort_size: int = 1
    candidate_window: int = 1
    max_consecutive: int = 1
    max_inflight_states: int = 8
    policy: str = "fifo"
    group_size: int = 1
    num_layers: Optional[int] = None
    slots_per_layer: int = 0
    max_replacements: int = 0
    frequency_plans: LayerPlans = ()
    expert_nbytes: int = 0
    h2d_expert_ms: float = 5.4
    d2d_expert_ms: float = 0.08
    route_entry_gain_ms: float = 0.0
    copy_contention_ms_per_expert: float = 0.0
    eviction_route_weight: float = 0.0
    queue_penalty_ms_per_s: float = 0.0
    min_gain_ms: float = 0.0
    confidence_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if self.cohort_size <= 0:
            raise ValueError("cohort_size must be positive")
        if self.candidate_window < self.cohort_size:
            raise ValueError("candidate_window must be at least cohort_size")
        if self.max_consecutive <= 0 or self.max_inflight_states <= 0:
            raise ValueError("fairness and inflight bounds must be positive")
        if self.policy not in {"fifo", "min_delta", "cost_oracle"}:
            raise ValueError(
                "stage policy must be one of fifo, min_delta, or cost_oracle"
            )
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        num_layers = (
            self.num_groups * self.group_size
            if self.num_layers is None
            else int(self.num_layers)
        )
        if num_layers <= 0 or (num_layers + self.group_size - 1) // self.group_size != self.num_groups:
            raise ValueError("num_layers/group_size do not match num_groups")
        object.__setattr__(self, "num_layers", num_layers)
        if self.slots_per_layer < 0:
            raise ValueError("slots_per_layer cannot be negative")
        if not 0 <= self.max_replacements <= self.slots_per_layer:
            raise ValueError("max_replacements must be in [0, slots_per_layer]")
        normalized_frequency = normalize_layer_plans(self.frequency_plans)
        object.__setattr__(self, "frequency_plans", normalized_frequency)
        if normalized_frequency:
            expected_layers = set(range(num_layers))
            actual_layers = {layer_idx for layer_idx, _ in normalized_frequency}
            if actual_layers != expected_layers:
                raise ValueError("frequency plans must cover every model layer")
            if self.slots_per_layer <= 0 or any(
                len(experts) != self.slots_per_layer
                for _, experts in normalized_frequency
            ):
                raise ValueError(
                    "frequency plan width must equal positive slots_per_layer"
                )
        elif self.policy != "fifo":
            raise ValueError("non-FIFO policies require frequency plans")
        if self.expert_nbytes < 0:
            raise ValueError("expert_nbytes cannot be negative")
        nonnegative_costs = {
            "h2d_expert_ms": self.h2d_expert_ms,
            "d2d_expert_ms": self.d2d_expert_ms,
            "route_entry_gain_ms": self.route_entry_gain_ms,
            "copy_contention_ms_per_expert": self.copy_contention_ms_per_expert,
            "eviction_route_weight": self.eviction_route_weight,
            "queue_penalty_ms_per_s": self.queue_penalty_ms_per_s,
        }
        if any(not math.isfinite(value) or value < 0 for value in nonnegative_costs.values()):
            raise ValueError(f"cost parameters must be finite and non-negative: {nonnegative_costs}")
        if not math.isfinite(self.min_gain_ms):
            raise ValueError("min_gain_ms must be finite")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


class StageReadyQueues:
    """Own all stage states and enforce single-owner transitions."""

    def __init__(self, *, num_groups: int, max_inflight_states: int):
        if num_groups <= 0 or max_inflight_states <= 0:
            raise ValueError("queue bounds must be positive")
        self.num_groups = int(num_groups)
        self.max_inflight_states = int(max_inflight_states)
        self.states: Dict[int, ChunkStageState] = {}
        self.ready_ids: Dict[int, List[int]] = {
            group_id: [] for group_id in range(self.num_groups)
        }
        self.active_request_ids: Dict[str, int] = {}
        self.queue_epoch = 0

    def _bump(self) -> None:
        self.queue_epoch += 1

    def _remove_ready_id(self, state: ChunkStageState) -> None:
        values = self.ready_ids[state.group_id]
        try:
            values.remove(state.state_id)
        except ValueError as exc:
            raise RuntimeError(
                f"READY state {state.state_id} is missing from group {state.group_id}"
            ) from exc

    def _append_ready_id(self, state: ChunkStageState) -> None:
        values = self.ready_ids[state.group_id]
        if state.state_id in values:
            raise RuntimeError(f"state {state.state_id} is already in a ready queue")
        values.append(state.state_id)
        values.sort(key=lambda state_id: self._ready_sort_key(self.states[state_id]))

    @staticmethod
    def _ready_sort_key(state: ChunkStageState) -> Tuple[int, int]:
        return state.enqueue_seq, state.state_id

    def admit(self, state: ChunkStageState) -> None:
        if state.state_id in self.states:
            raise RuntimeError(f"duplicate stage state ID: {state.state_id}")
        if state.request_id in self.active_request_ids:
            raise RuntimeError(
                "M20-B v1 allows only one in-flight chunk per request: "
                f"{state.request_id}"
            )
        if len(self.active_request_ids) >= self.max_inflight_states:
            raise RuntimeError("stage-ready inflight state limit reached")
        if state.status != StageState.NEW:
            raise RuntimeError("only NEW states can be admitted")
        state.status = StageState.READY
        state.state_version += 1
        self.states[state.state_id] = state
        self.active_request_ids[state.request_id] = state.state_id
        self._append_ready_id(state)
        self._bump()

    def ready_groups(self) -> List[int]:
        return [group_id for group_id, values in self.ready_ids.items() if values]

    def ready_states(self, group_id: int, limit: Optional[int] = None) -> List[ChunkStageState]:
        if not 0 <= group_id < self.num_groups:
            raise IndexError(f"group ID out of range: {group_id}")
        ids = self.ready_ids[group_id]
        if limit is not None:
            ids = ids[: max(0, int(limit))]
        return [self.states[state_id] for state_id in ids]

    def oldest_expired(self, now: float) -> Optional[ChunkStageState]:
        values = [
            state
            for group_id in self.ready_groups()
            for state in self.ready_states(group_id)
            if state.deadline_at <= now
        ]
        return min(
            values,
            key=lambda state: (state.deadline_at, state.enqueue_seq, state.state_id),
            default=None,
        )

    def oldest_ready(self, groups: Optional[Iterable[int]] = None) -> Optional[ChunkStageState]:
        group_ids = self.ready_groups() if groups is None else list(groups)
        values = [self.ready_states(group_id, 1)[0] for group_id in group_ids if self.ready_ids[group_id]]
        return min(values, key=self._ready_sort_key, default=None)

    def reserve(self, state_ids: Sequence[int], ticket_id: int) -> Tuple[int, ...]:
        if not state_ids or len(set(state_ids)) != len(state_ids):
            raise RuntimeError("reservation requires unique non-empty state IDs")
        states = [self.states[int(state_id)] for state_id in state_ids]
        group_ids = {state.group_id for state in states}
        request_ids = {state.request_id for state in states}
        if len(group_ids) != 1 or len(request_ids) != len(states):
            raise RuntimeError("a reservation requires one group and distinct requests")
        if any(state.status != StageState.READY for state in states):
            raise RuntimeError("only READY states can be reserved")
        for state in states:
            self._remove_ready_id(state)
            state.status = StageState.RESERVED
            state.reserved_ticket_id = int(ticket_id)
            state.state_version += 1
        self._bump()
        return tuple(state.state_version for state in states)

    def mark_running(self, ticket: NextActionTicket) -> None:
        states = [self.states[state_id] for state_id in ticket.state_ids]
        for state, expected_version in zip(states, ticket.state_versions):
            if (
                state.status != StageState.RESERVED
                or state.reserved_ticket_id != ticket.ticket_id
                or state.state_version != expected_version
            ):
                raise RuntimeError(
                    f"ticket {ticket.ticket_id} has stale reservation for state "
                    f"{state.state_id}"
                )
        for state in states:
            state.status = StageState.RUNNING
            state.state_version += 1
        self._bump()

    def unreserve(self, ticket: NextActionTicket, *, now: Optional[float] = None) -> None:
        ready_time = time.monotonic() if now is None else float(now)
        states = [self.states[state_id] for state_id in ticket.state_ids]
        if any(
            state.status != StageState.RESERVED
            or state.reserved_ticket_id != ticket.ticket_id
            for state in states
        ):
            raise RuntimeError("only the owning ticket can release a reservation")
        for state in states:
            state.status = StageState.READY
            state.reserved_ticket_id = None
            state.ready_since = ready_time
            state.state_version += 1
            self._append_ready_id(state)
        self._bump()

    def complete_group(self, ticket: NextActionTicket, *, now: Optional[float] = None) -> bool:
        ready_time = time.monotonic() if now is None else float(now)
        states = [self.states[state_id] for state_id in ticket.state_ids]
        if any(
            state.status != StageState.RUNNING
            or state.reserved_ticket_id != ticket.ticket_id
            or state.group_id != ticket.group_id
            for state in states
        ):
            raise RuntimeError("ticket completion does not own all running states")

        final = ticket.group_id == self.num_groups - 1
        for state in states:
            state.state_version += 1
            state.reserved_ticket_id = None
            if final:
                state.status = StageState.FINALIZING
            else:
                state.group_id += 1
                state.status = StageState.READY
                state.ready_since = ready_time
                self._append_ready_id(state)
        self._bump()
        return final

    def finalize(self, state_ids: Sequence[int]) -> None:
        states = [self.states[int(state_id)] for state_id in state_ids]
        if any(state.status != StageState.FINALIZING for state in states):
            raise RuntimeError("only FINALIZING states can become DONE")
        for state in states:
            state.status = StageState.DONE
            state.state_version += 1
            self.active_request_ids.pop(state.request_id, None)
        self._bump()

    def fail(self, state_ids: Sequence[int], reason: str) -> None:
        for raw_state_id in state_ids:
            state = self.states[int(raw_state_id)]
            if state.status == StageState.READY:
                self._remove_ready_id(state)
            if state.status in {StageState.DONE, StageState.FAILED}:
                raise RuntimeError(f"state {state.state_id} is already terminal")
            state.status = StageState.FAILED
            state.failure_reason = str(reason)
            state.reserved_ticket_id = None
            state.state_version += 1
            self.active_request_ids.pop(state.request_id, None)
        self._bump()

    def assert_invariants(self) -> None:
        queued = [state_id for values in self.ready_ids.values() for state_id in values]
        if len(queued) != len(set(queued)):
            raise RuntimeError("a state appears in more than one ready queue")
        for state_id, state in self.states.items():
            in_ready = state_id in queued
            if in_ready != (state.status == StageState.READY):
                raise RuntimeError(
                    f"state {state_id} READY membership/status mismatch: {state.status}"
                )
            if in_ready and state_id not in self.ready_ids[state.group_id]:
                raise RuntimeError(f"state {state_id} is queued under the wrong group")
            if state.status in {StageState.RESERVED, StageState.RUNNING}:
                if state.reserved_ticket_id is None:
                    raise RuntimeError(f"owned state {state_id} has no ticket")
            elif state.reserved_ticket_id is not None:
                raise RuntimeError(f"unowned state {state_id} retains a ticket")
        expected_active = {
            state.request_id: state.state_id
            for state in self.states.values()
            if state.status not in {StageState.DONE, StageState.FAILED}
        }
        if expected_active != self.active_request_ids:
            raise RuntimeError("active request ownership map is inconsistent")


PlanBuilder = Callable[
    [int, Sequence[ChunkStageState]],
    Mapping[int, Sequence[int]] | Iterable[Tuple[int, Sequence[int]]],
]


@dataclass
class _AbstractBufferRecord:
    buffer_id: int
    group_id: Optional[int] = None
    version: int = 0
    content_version: int = 0
    plans: Dict[int, Tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ActionCandidate:
    group_id: int
    states: Tuple[ChunkStageState, ...]
    layer_plans: LayerPlans
    logical_source_plans: LayerPlans
    logical_source_buffer_id: Optional[int]
    confidence: float
    target_buffer_id: Optional[int]
    expected_buffer_versions: Tuple[int, int]
    copy_ops: Tuple[CopyOp, ...]
    score: ScoreBreakdown
    replacements: int
    covered_routes: float
    fallback: str

    @property
    def token_count(self) -> int:
        return sum(state.token_count for state in self.states)


class StageScheduler:
    """Deterministic stage selector with bounded reuse and cost-aware plans.

    Physical full/delta materialization remains worker-owned.  The scheduler
    keeps a separate two-buffer delta model solely for deterministic candidate
    comparison, so a full reference run can freeze plans that delta replay can
    execute without inheriting the full path's target-buffer choices.
    """

    def __init__(
        self,
        config: StageSchedulerConfig,
        *,
        plan_builder: Optional[PlanBuilder] = None,
        provider_version: str = "",
        trace_writer: Optional["ActionTraceWriter"] = None,
    ):
        self.config = config
        self.queues = StageReadyQueues(
            num_groups=config.num_groups,
            max_inflight_states=config.max_inflight_states,
        )
        self.plan_builder = plan_builder or (lambda _group_id, _states: {})
        self.provider_version = str(provider_version)
        self.trace_writer = trace_writer
        self.next_ticket_id = 0
        self.last_group_id: Optional[int] = None
        self.same_stage_streak = 0
        self._frequency_plans = dict(config.frequency_plans)
        self._logical_plans_by_group: Dict[int, Dict[int, Tuple[int, ...]]] = {}
        if self._frequency_plans:
            for group_id in range(config.num_groups):
                self._logical_plans_by_group[group_id] = {
                    layer_idx: self._frequency_plans[layer_idx]
                    for layer_idx in self._group_layers(group_id)
                }
        self._abstract_buffers = [_AbstractBufferRecord(0), _AbstractBufferRecord(1)]
        self._abstract_content_version = 0
        self._pending_candidates: Dict[int, _ActionCandidate] = {}

    @property
    def _working_set_enabled(self) -> bool:
        return bool(self._frequency_plans)

    def _group_layers(self, group_id: int) -> range:
        start = int(group_id) * self.config.group_size
        return range(start, min(start + self.config.group_size, int(self.config.num_layers)))

    def _current_plans(self, group_id: int) -> Dict[int, Tuple[int, ...]]:
        physical_source = self._latest_buffer_for_group(int(group_id))
        if physical_source is not None:
            return dict(physical_source.plans)
        plans = self._logical_plans_by_group.get(int(group_id))
        if plans is None:
            raise RuntimeError(f"group {group_id} has no frequency/current plan")
        return dict(plans)

    def _buffers_for_group(self, group_id: int) -> List[_AbstractBufferRecord]:
        return sorted(
            [record for record in self._abstract_buffers if record.group_id == group_id],
            key=lambda record: (record.content_version, record.version, record.buffer_id),
            reverse=True,
        )

    def _latest_buffer_for_group(
        self, group_id: int
    ) -> Optional[_AbstractBufferRecord]:
        records = self._buffers_for_group(group_id)
        return records[0] if records else None

    def _exact_buffer_for_plan(
        self, group_id: int, plans: Mapping[int, Sequence[int]]
    ) -> Optional[_AbstractBufferRecord]:
        normalized = {
            int(layer_idx): tuple(int(expert) for expert in experts)
            for layer_idx, experts in plans.items()
        }
        return next(
            (
                record
                for record in self._buffers_for_group(group_id)
                if record.plans == normalized
            ),
            None,
        )

    def _target_buffer_for_group(
        self, group_id: int
    ) -> Tuple[_AbstractBufferRecord, Optional[_AbstractBufferRecord]]:
        source = self._latest_buffer_for_group(group_id)
        target_id = group_id % 2 if source is None else 1 - source.buffer_id
        return self._abstract_buffers[target_id], source

    @staticmethod
    def _expert_slot(
        record: Optional[_AbstractBufferRecord], layer_idx: int, expert_id: int
    ) -> Optional[int]:
        if record is None:
            return None
        try:
            return record.plans.get(layer_idx, ()).index(int(expert_id))
        except ValueError:
            return None

    def _assign_slots(
        self,
        group_id: int,
        layer_idx: int,
        selected: Iterable[int],
        preferred_order: Sequence[int],
    ) -> Tuple[int, ...]:
        selected_set = {int(expert) for expert in selected}
        if len(selected_set) != self.config.slots_per_layer:
            raise RuntimeError(
                f"layer {layer_idx} selected {len(selected_set)} experts, "
                f"expected {self.config.slots_per_layer}"
            )
        current = self._current_plans(group_id)[layer_idx]
        if selected_set == set(current):
            return current

        target, source = self._target_buffer_for_group(group_id)
        assigned: List[Optional[int]] = [None] * self.config.slots_per_layer
        used = set()
        if target.group_id == group_id:
            for slot, expert in enumerate(target.plans.get(layer_idx, ())):
                if expert in selected_set and expert not in used:
                    assigned[slot] = expert
                    used.add(expert)
        if source is not None:
            for slot, expert in enumerate(source.plans.get(layer_idx, ())):
                if (
                    slot < len(assigned)
                    and assigned[slot] is None
                    and expert in selected_set
                    and expert not in used
                ):
                    assigned[slot] = expert
                    used.add(expert)
        remaining = [
            int(expert)
            for expert in preferred_order
            if int(expert) in selected_set and int(expert) not in used
        ]
        remaining.extend(sorted(selected_set - used - set(remaining)))
        iterator = iter(remaining)
        result = tuple(
            next(iterator) if expert is None else expert for expert in assigned
        )
        if len(result) != len(set(result)):
            raise RuntimeError(f"slot assignment duplicated an expert: {result}")
        return result

    def _simulate_materialization(
        self, group_id: int, plans: Mapping[int, Sequence[int]]
    ) -> Tuple[int, Tuple[int, int], Tuple[CopyOp, ...]]:
        normalized = {
            int(layer_idx): tuple(int(expert) for expert in experts)
            for layer_idx, experts in plans.items()
        }
        versions = tuple(record.version for record in self._abstract_buffers)
        exact = self._exact_buffer_for_plan(group_id, normalized)
        if exact is not None:
            return exact.buffer_id, versions, ()

        target, source = self._target_buffer_for_group(group_id)
        ops: List[CopyOp] = []
        for layer_idx in self._group_layers(group_id):
            for dst_slot, expert_id in enumerate(normalized[layer_idx]):
                target_slot = (
                    self._expert_slot(target, layer_idx, expert_id)
                    if target.group_id == group_id
                    else None
                )
                if target_slot == dst_slot:
                    ops.append(
                        CopyOp(
                            kind=CopyKind.RETAIN,
                            layer_idx=layer_idx,
                            expert_id=expert_id,
                            dst_buffer_id=target.buffer_id,
                            dst_slot=dst_slot,
                            nbytes=self.config.expert_nbytes,
                            src_buffer_id=target.buffer_id,
                            src_slot=dst_slot,
                        )
                    )
                    continue
                source_slot = self._expert_slot(source, layer_idx, expert_id)
                if source is not None and source_slot is not None:
                    ops.append(
                        CopyOp(
                            kind=CopyKind.D2D,
                            layer_idx=layer_idx,
                            expert_id=expert_id,
                            dst_buffer_id=target.buffer_id,
                            dst_slot=dst_slot,
                            nbytes=self.config.expert_nbytes,
                            src_buffer_id=source.buffer_id,
                            src_slot=source_slot,
                        )
                    )
                else:
                    ops.append(
                        CopyOp(
                            kind=CopyKind.H2D,
                            layer_idx=layer_idx,
                            expert_id=expert_id,
                            dst_buffer_id=target.buffer_id,
                            dst_slot=dst_slot,
                            nbytes=self.config.expert_nbytes,
                        )
                    )
        return target.buffer_id, versions, tuple(ops)

    def _aggregate_demand(
        self, group_id: int, states: Sequence[ChunkStageState]
    ) -> Tuple[Dict[int, Dict[int, float]], float]:
        demand = {layer_idx: {} for layer_idx in self._group_layers(group_id)}
        confidence_values: List[float] = []
        for state in states:
            state_demand = state.demand_by_layer or {}
            state_confidence = state.confidence_by_layer or {}
            for layer_idx in self._group_layers(group_id):
                layer_demand = state_demand.get(layer_idx)
                confidence_values.append(
                    float(
                        state_confidence.get(
                            layer_idx, 1.0 if layer_demand is not None else 0.0
                        )
                    )
                )
                if layer_demand is None:
                    continue
                for raw_expert, raw_value in layer_demand.items():
                    expert_id = int(raw_expert)
                    value = float(raw_value)
                    if expert_id < 0 or not math.isfinite(value) or value < 0:
                        raise ValueError("stage demand must be finite and non-negative")
                    if value > 0:
                        demand[layer_idx][expert_id] = (
                            demand[layer_idx].get(expert_id, 0.0) + value
                        )
        confidence = min(confidence_values, default=0.0)
        if not math.isfinite(confidence):
            raise ValueError("stage demand confidence must be finite")
        return demand, min(1.0, max(0.0, confidence))

    @staticmethod
    def _covered_routes(
        plans: Mapping[int, Sequence[int]],
        demand: Mapping[int, Mapping[int, float]],
    ) -> float:
        return sum(
            float(demand.get(layer_idx, {}).get(expert_id, 0.0))
            for layer_idx, experts in plans.items()
            for expert_id in experts
        )

    def _eviction_loss_ms(
        self,
        group_id: int,
        plans: Mapping[int, Sequence[int]],
        selected_states: Sequence[ChunkStageState],
    ) -> float:
        if self.config.eviction_route_weight <= 0 or self.config.route_entry_gain_ms <= 0:
            return 0.0
        selected_ids = {state.state_id for state in selected_states}
        future = [
            state
            for state in self.queues.ready_states(
                group_id, self.config.candidate_window
            )
            if state.state_id not in selected_ids
        ]
        current = self._current_plans(group_id)
        lost_routes = 0.0
        for layer_idx in self._group_layers(group_id):
            evicted = set(current[layer_idx]) - set(plans[layer_idx])
            for state in future:
                layer_demand = (state.demand_by_layer or {}).get(layer_idx, {})
                lost_routes += sum(float(layer_demand.get(expert, 0.0)) for expert in evicted)
        return (
            lost_routes
            * self.config.route_entry_gain_ms
            * self.config.eviction_route_weight
        )

    def _queue_penalty_ms(
        self, states: Sequence[ChunkStageState], now: float
    ) -> float:
        if self.config.queue_penalty_ms_per_s <= 0:
            return 0.0
        selected_ids = {state.state_id for state in states}
        anchor_seq = min(state.enqueue_seq for state in states)
        skipped_wait_s = sum(
            max(0.0, now - state.ready_since)
            for group_id in self.queues.ready_groups()
            for state in self.queues.ready_states(group_id)
            if state.state_id not in selected_ids and state.enqueue_seq < anchor_seq
        )
        return skipped_wait_s * self.config.queue_penalty_ms_per_s

    def _score_plans(
        self,
        group_id: int,
        states: Sequence[ChunkStageState],
        plans: Mapping[int, Sequence[int]],
        demand: Mapping[int, Mapping[int, float]],
        now: float,
    ) -> Tuple[ScoreBreakdown, int, Tuple[int, int], Tuple[CopyOp, ...]]:
        target, versions, ops = self._simulate_materialization(group_id, plans)
        h2d_count = sum(op.kind == CopyKind.H2D for op in ops)
        d2d_count = sum(op.kind == CopyKind.D2D for op in ops)
        copied_count = h2d_count + d2d_count
        covered_route_entries = self._covered_routes(plans, demand)
        score = ScoreBreakdown(
            compute_gain_ms=(
                covered_route_entries * self.config.route_entry_gain_ms
            ),
            materialization_ms=(
                h2d_count * self.config.h2d_expert_ms
                + d2d_count * self.config.d2d_expert_ms
            ),
            copy_contention_ms=(
                copied_count * self.config.copy_contention_ms_per_expert
            ),
            eviction_loss_ms=self._eviction_loss_ms(group_id, plans, states),
            queue_penalty_ms=self._queue_penalty_ms(states, now),
            covered_route_entries=covered_route_entries,
        )
        return score, target, versions, ops

    def _build_rank_bounded_plans(
        self,
        group_id: int,
        demand: Mapping[int, Mapping[int, float]],
    ) -> Dict[int, Tuple[int, ...]]:
        plans = self._current_plans(group_id)
        max_replacements = self.config.max_replacements
        if max_replacements <= 0:
            return plans
        for layer_idx in self._group_layers(group_id):
            active = plans[layer_idx]
            scores = demand.get(layer_idx, {})
            new_experts = sorted(
                (expert for expert, value in scores.items() if value > 0 and expert not in active),
                key=lambda expert: (-float(scores[expert]), int(expert)),
            )
            victims = sorted(
                active,
                key=lambda expert: (
                    float(scores.get(expert, 0.0)),
                    -active.index(expert),
                    int(expert),
                ),
            )
            selected = set(active)
            preferred = list(active)
            accepted = 0
            for new_expert, victim in zip(new_experts, victims):
                if accepted >= max_replacements:
                    break
                if float(scores.get(new_expert, 0.0)) <= float(scores.get(victim, 0.0)):
                    continue
                selected.remove(victim)
                selected.add(new_expert)
                preferred[preferred.index(victim)] = new_expert
                accepted += 1
            plans[layer_idx] = self._assign_slots(
                group_id, layer_idx, selected, preferred
            )
        return plans

    def _build_cost_bounded_plans(
        self,
        group_id: int,
        states: Sequence[ChunkStageState],
        demand: Mapping[int, Mapping[int, float]],
        now: float,
    ) -> Dict[int, Tuple[int, ...]]:
        plans = self._current_plans(group_id)
        if self.config.max_replacements <= 0:
            return plans
        original = self._current_plans(group_id)
        for layer_idx in self._group_layers(group_id):
            scores = demand.get(layer_idx, {})
            candidate_limit = max(
                self.config.slots_per_layer, 2 * self.config.max_replacements
            )
            new_experts = sorted(
                (
                    expert
                    for expert, value in scores.items()
                    if value > 0 and expert not in original[layer_idx]
                ),
                key=lambda expert: (-float(scores[expert]), int(expert)),
            )[:candidate_limit]
            for _ in range(self.config.max_replacements):
                base_score = self._score_plans(
                    group_id, states, plans, demand, now
                )[0].net_gain_ms
                active = plans[layer_idx]
                changed = set(active) - set(original[layer_idx])
                if len(changed) >= self.config.max_replacements:
                    break
                best: Optional[Tuple[float, float, int, int, Tuple[int, ...]]] = None
                for new_expert in new_experts:
                    if new_expert in active:
                        continue
                    for victim in active:
                        if victim not in original[layer_idx]:
                            continue
                        if float(scores.get(new_expert, 0.0)) <= float(
                            scores.get(victim, 0.0)
                        ):
                            continue
                        selected = set(active)
                        selected.remove(victim)
                        selected.add(new_expert)
                        preferred = [
                            new_expert if expert == victim else expert
                            for expert in active
                        ]
                        layer_plan = self._assign_slots(
                            group_id, layer_idx, selected, preferred
                        )
                        trial = dict(plans)
                        trial[layer_idx] = layer_plan
                        trial_score = self._score_plans(
                            group_id, states, trial, demand, now
                        )[0].net_gain_ms
                        item = (
                            trial_score - base_score,
                            float(scores.get(new_expert, 0.0))
                            - float(scores.get(victim, 0.0)),
                            -int(new_expert),
                            -int(victim),
                            layer_plan,
                        )
                        if best is None or item > best:
                            best = item
                if best is None or best[0] <= 0:
                    break
                plans[layer_idx] = best[-1]
        return plans

    def _make_candidate(
        self,
        group_id: int,
        states: Sequence[ChunkStageState],
        *,
        now: float,
        force_current: bool = False,
        fallback: str = "",
    ) -> _ActionCandidate:
        normalized_states = tuple(states)
        if not normalized_states:
            raise RuntimeError("cannot build an empty stage candidate")
        if not self._working_set_enabled:
            plans = normalize_layer_plans(self.plan_builder(group_id, normalized_states))
            return _ActionCandidate(
                group_id=group_id,
                states=normalized_states,
                layer_plans=plans,
                logical_source_plans=(),
                logical_source_buffer_id=None,
                confidence=1.0,
                target_buffer_id=None,
                expected_buffer_versions=(-1, -1),
                copy_ops=(),
                score=ScoreBreakdown(),
                replacements=0,
                covered_routes=0.0,
                fallback=fallback,
            )

        demand, confidence = self._aggregate_demand(group_id, normalized_states)
        logical_source = self._current_plans(group_id)
        source_record = self._latest_buffer_for_group(group_id)
        low_confidence = confidence < self.config.confidence_threshold
        if force_current or low_confidence:
            plans_dict = self._current_plans(group_id)
            if low_confidence and not fallback:
                fallback = "low_confidence_fifo_current"
        elif self.config.policy == "cost_oracle":
            plans_dict = self._build_cost_bounded_plans(
                group_id, normalized_states, demand, now
            )
        else:
            plans_dict = self._build_rank_bounded_plans(group_id, demand)
        score, target, versions, ops = self._score_plans(
            group_id, normalized_states, plans_dict, demand, now
        )
        current = self._current_plans(group_id)
        replacements = sum(
            len(set(plans_dict[layer_idx]) - set(current[layer_idx]))
            for layer_idx in self._group_layers(group_id)
        )
        return _ActionCandidate(
            group_id=group_id,
            states=normalized_states,
            layer_plans=normalize_layer_plans(plans_dict),
            logical_source_plans=normalize_layer_plans(logical_source),
            logical_source_buffer_id=(
                None if source_record is None else source_record.buffer_id
            ),
            confidence=confidence,
            target_buffer_id=target,
            expected_buffer_versions=versions,
            copy_ops=ops,
            score=score,
            replacements=replacements,
            covered_routes=self._covered_routes(plans_dict, demand),
            fallback=fallback,
        )

    def _eligible_groups(self) -> List[int]:
        groups = self.queues.ready_groups()
        if (
            self.last_group_id in groups
            and self.same_stage_streak >= self.config.max_consecutive
            and len(groups) > 1
        ):
            groups.remove(int(self.last_group_id))
        return groups

    def _fifo_anchor(self, eligible_groups: Sequence[int]) -> ChunkStageState:
        reuse_last_group = (
            self.last_group_id in eligible_groups
            and self.same_stage_streak < self.config.max_consecutive
        )
        groups = [int(self.last_group_id)] if reuse_last_group else list(eligible_groups)
        anchor = self.queues.oldest_ready(groups)
        if anchor is None:
            raise RuntimeError("non-empty eligible groups have no READY anchor")
        return anchor

    def _cohort_with_anchor(
        self, anchor: ChunkStageState
    ) -> Tuple[ChunkStageState, ...]:
        window = self.queues.ready_states(
            anchor.group_id, self.config.candidate_window
        )
        others = [state for state in window if state.state_id != anchor.state_id]
        return tuple([anchor, *others[: self.config.cohort_size - 1]])

    def _policy_candidates(
        self, eligible_groups: Sequence[int], now: float
    ) -> List[_ActionCandidate]:
        candidates: List[_ActionCandidate] = []
        for group_id in eligible_groups:
            window = self.queues.ready_states(
                group_id, self.config.candidate_window
            )
            anchor = window[0]
            max_size = min(self.config.cohort_size, len(window))
            sizes = (
                [max_size]
                if self.config.policy == "min_delta"
                else list(range(max_size, 0, -1))
            )
            for size in sizes:
                for partners in combinations(window[1:], size - 1):
                    candidates.append(
                        self._make_candidate(
                            group_id,
                            (anchor, *partners),
                            now=now,
                        )
                    )
        return candidates

    @staticmethod
    def _candidate_identity(candidate: _ActionCandidate) -> Tuple[Any, ...]:
        return (
            min(state.enqueue_seq for state in candidate.states),
            tuple(state.state_id for state in candidate.states),
            compute_plan_hash(candidate.group_id, candidate.layer_plans),
        )

    def _select_candidate(
        self, eligible_groups: Sequence[int], now: float
    ) -> _ActionCandidate:
        candidates = self._policy_candidates(eligible_groups, now)
        if not candidates:
            raise RuntimeError("eligible stage queues produced no candidates")
        if self.config.policy == "min_delta":
            def min_delta_key(candidate: _ActionCandidate) -> Tuple[Any, ...]:
                tokens = max(1, candidate.token_count)
                h2d = sum(op.kind == CopyKind.H2D for op in candidate.copy_ops)
                d2d = sum(op.kind == CopyKind.D2D for op in candidate.copy_ops)
                return (
                    h2d / tokens,
                    candidate.replacements / tokens,
                    d2d / tokens,
                    -candidate.covered_routes / tokens,
                    -len(candidate.states),
                    self._candidate_identity(candidate),
                )

            return min(candidates, key=min_delta_key)

        best = min(
            candidates,
            key=lambda candidate: (
                -candidate.score.net_gain_ms,
                -candidate.covered_routes,
                -len(candidate.states),
                self._candidate_identity(candidate),
            ),
        )
        if best.score.net_gain_ms > self.config.min_gain_ms:
            return best

        anchor = self._fifo_anchor(eligible_groups)
        return self._make_candidate(
            anchor.group_id,
            self._cohort_with_anchor(anchor),
            now=now,
            force_current=True,
            fallback="nonpositive_fifo_current",
        )

    def _reserve_candidate(self, candidate: _ActionCandidate) -> NextActionTicket:
        ticket_id = self.next_ticket_id
        states = candidate.states
        state_versions = self.queues.reserve(
            [state.state_id for state in states], ticket_id
        )
        try:
            ticket = NextActionTicket.create(
                ticket_id=ticket_id,
                queue_epoch=self.queues.queue_epoch,
                group_id=candidate.group_id,
                states=states,
                state_versions=state_versions,
                layer_plans=candidate.layer_plans,
                policy=self.config.policy,
                provider_version=self.provider_version,
                confidence=candidate.confidence,
                logical_source_plans=candidate.logical_source_plans,
                logical_source_buffer_id=candidate.logical_source_buffer_id,
                target_buffer_id=candidate.target_buffer_id,
                expected_buffer_versions=candidate.expected_buffer_versions,
                copy_ops=candidate.copy_ops,
                score=candidate.score,
                fallback=candidate.fallback,
            )
        except Exception:
            for state in states:
                state.status = StageState.READY
                state.reserved_ticket_id = None
                state.state_version += 1
                self.queues._append_ready_id(state)
            self.queues._bump()
            raise
        self._pending_candidates[ticket_id] = candidate
        self.next_ticket_id += 1
        if self.trace_writer is not None:
            self.trace_writer.append_event("planned", ticket=ticket)
        return ticket

    def choose_next(self, *, now: Optional[float] = None) -> Optional[NextActionTicket]:
        if self._pending_candidates:
            raise RuntimeError("M20-B v1 permits only one outstanding action ticket")
        current_time = time.monotonic() if now is None else float(now)
        ready_groups = self.queues.ready_groups()
        if not ready_groups:
            return None

        expired = self.queues.oldest_expired(current_time)
        if expired is not None:
            candidate = self._make_candidate(
                expired.group_id,
                self._cohort_with_anchor(expired),
                now=current_time,
                force_current=True,
                fallback="deadline_fifo",
            )
            return self._reserve_candidate(candidate)

        eligible_groups = self._eligible_groups()
        if self.config.policy == "fifo":
            anchor = self._fifo_anchor(eligible_groups)
            candidate = self._make_candidate(
                anchor.group_id,
                self._cohort_with_anchor(anchor),
                now=current_time,
                fallback="fifo",
            )
        else:
            candidate = self._select_candidate(eligible_groups, current_time)
        return self._reserve_candidate(candidate)

    def mark_running(self, ticket: NextActionTicket) -> None:
        self.queues.mark_running(ticket)
        if ticket.group_id == self.last_group_id:
            self.same_stage_streak += 1
        else:
            self.last_group_id = ticket.group_id
            self.same_stage_streak = 1
        if self.trace_writer is not None:
            self.trace_writer.append_event("dispatched", ticket=ticket)

    def materialize_ticket(
        self,
        ticket: NextActionTicket,
        layer_plans: Mapping[int, Sequence[int]]
        | Iterable[Tuple[int, Sequence[int]]],
        *,
        plan_hash: Optional[str] = None,
        target_buffer_id: Optional[int] = None,
        expected_buffer_versions: Optional[Sequence[int]] = None,
        copy_ops: Optional[Sequence[CopyOp]] = None,
    ) -> NextActionTicket:
        states = [self.queues.states[state_id] for state_id in ticket.state_ids]
        if any(
            state.status != StageState.RUNNING
            or state.reserved_ticket_id != ticket.ticket_id
            for state in states
        ):
            raise RuntimeError(
                f"ticket {ticket.ticket_id} cannot be materialized without RUNNING ownership"
            )
        materialized = ticket.materialize(
            layer_plans,
            plan_hash=plan_hash,
            target_buffer_id=target_buffer_id,
            expected_buffer_versions=expected_buffer_versions,
            copy_ops=copy_ops,
        )
        if self.trace_writer is not None:
            self.trace_writer.append_event("materialized", ticket=materialized)
        return materialized

    def complete_group(
        self, ticket: NextActionTicket, *, now: Optional[float] = None
    ) -> bool:
        if self._working_set_enabled and ticket.layer_plans:
            plans = ticket.plan_dict
            expected_source = dict(ticket.logical_source_plans)
            current_source = self._current_plans(ticket.group_id)
            if expected_source and expected_source != current_source:
                raise RuntimeError(
                    f"ticket {ticket.ticket_id} logical source diverged before commit"
                )
            exact = self._exact_buffer_for_plan(ticket.group_id, plans)
            if exact is None:
                target, _source = self._target_buffer_for_group(ticket.group_id)
                target.version += 1
                self._abstract_content_version += 1
                target.content_version = self._abstract_content_version
                target.group_id = ticket.group_id
                target.plans = dict(plans)
            self._logical_plans_by_group[ticket.group_id] = dict(plans)
        final = self.queues.complete_group(ticket, now=now)
        self._pending_candidates.pop(ticket.ticket_id, None)
        if self.trace_writer is not None:
            self.trace_writer.append_event(
                "completed", ticket=ticket, details={"final_group": final}
            )
        return final

    def unreserve(
        self, ticket: NextActionTicket, *, now: Optional[float] = None
    ) -> None:
        self.queues.unreserve(ticket, now=now)
        self._pending_candidates.pop(ticket.ticket_id, None)
        if self.trace_writer is not None:
            self.trace_writer.append_event("unreserved", ticket=ticket)


class ActionTraceWriter:
    """Write a hash-chained JSONL trace suitable for exact sync replay."""

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        overwrite: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"action trace already exists: {self.path}")
        self._handle = self.path.open("w", encoding="utf-8")
        self._sequence = 0
        self._previous_hash = ""
        self._write_record(
            {
                "record_type": "header",
                "schema_version": ACTION_TRACE_SCHEMA_VERSION,
                "metadata": dict(metadata or {}),
            }
        )

    def _write_record(self, payload: Mapping[str, Any]) -> None:
        record = {
            "sequence": self._sequence,
            "previous_hash": self._previous_hash,
            **dict(payload),
        }
        record_hash = sha256_json(record)
        stored = {**record, "record_hash": record_hash}
        self._handle.write(canonical_json(stored) + "\n")
        self._handle.flush()
        self._previous_hash = record_hash
        self._sequence += 1

    def append_event(
        self,
        event: str,
        *,
        ticket: NextActionTicket,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._write_record(
            {
                "record_type": "action",
                "schema_version": ACTION_TRACE_SCHEMA_VERSION,
                "event": str(event),
                "ticket": ticket.to_dict(),
                "details": dict(details or {}),
            }
        )

    @property
    def trace_hash(self) -> str:
        return self._previous_hash

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "ActionTraceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class ActionTrace:
    path: Path
    metadata: Mapping[str, Any]
    records: Tuple[Mapping[str, Any], ...]
    trace_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "ActionTrace":
        trace_path = Path(path)
        records: List[Mapping[str, Any]] = []
        previous_hash = ""
        with trace_path.open("r", encoding="utf-8") as handle:
            for expected_sequence, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                stored = json.loads(line)
                record_hash = str(stored.pop("record_hash", ""))
                if int(stored.get("sequence", -1)) != expected_sequence:
                    raise ValueError("action trace sequence is not contiguous")
                if stored.get("previous_hash", "") != previous_hash:
                    raise ValueError("action trace hash chain is broken")
                expected_hash = sha256_json(stored)
                if record_hash != expected_hash:
                    raise ValueError("action trace record hash mismatch")
                previous_hash = record_hash
                records.append({**stored, "record_hash": record_hash})
        if not records or records[0].get("record_type") != "header":
            raise ValueError("action trace is missing its header")
        if int(records[0].get("schema_version", -1)) != ACTION_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported action trace schema")
        for record in records[1:]:
            if int(record.get("schema_version", -1)) != ACTION_TRACE_SCHEMA_VERSION:
                raise ValueError("mixed action trace schema versions")
        return cls(
            path=trace_path,
            metadata=dict(records[0].get("metadata") or {}),
            records=tuple(records),
            trace_hash=previous_hash,
        )

    def iter_tickets(self, event: str = "planned") -> Iterator[NextActionTicket]:
        for record in self.records[1:]:
            if record.get("record_type") != "action" or record.get("event") != event:
                continue
            ticket = record.get("ticket")
            if not isinstance(ticket, Mapping):
                raise ValueError("action trace record has no valid ticket")
            yield NextActionTicket.from_dict(ticket)


class ActionReplayCursor:
    """Reserve exactly the states and plans recorded in a planned-action trace."""

    def __init__(self, trace: ActionTrace):
        self.trace = trace
        planned = tuple(trace.iter_tickets("planned"))
        materialized = tuple(trace.iter_tickets("materialized"))
        if materialized:
            planned_ids = tuple(ticket.ticket_id for ticket in planned)
            materialized_ids = tuple(ticket.ticket_id for ticket in materialized)
            if materialized_ids != planned_ids:
                raise ValueError(
                    "materialized action trace is incomplete or out of order: "
                    f"planned={planned_ids} materialized={materialized_ids}"
                )
            if any(not ticket.layer_plans for ticket in materialized):
                raise ValueError("materialized replay ticket has no layer plans")
            self.tickets = materialized
            self.ticket_event = "materialized"
        else:
            # Compatibility with B0a traces produced before placement binding.
            self.tickets = planned
            self.ticket_event = "planned"
        self.index = 0

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.tickets)

    def peek_next(self) -> Optional[NextActionTicket]:
        """Return the next frozen ticket without advancing replay ownership."""

        if self.exhausted:
            return None
        return self.tickets[self.index]

    def reserve_next(self, queues: StageReadyQueues) -> Optional[NextActionTicket]:
        if self.exhausted:
            return None
        expected = self.tickets[self.index]
        missing = [
            state_id
            for state_id in expected.state_ids
            if state_id not in queues.states
        ]
        if missing:
            max_admitted = max(queues.states, default=-1)
            if all(state_id > max_admitted for state_id in missing):
                return None
            raise RuntimeError(
                f"replay ticket {expected.ticket_id} references missing states "
                f"{missing} after state {max_admitted} was admitted"
            )
        states = [queues.states[state_id] for state_id in expected.state_ids]
        for offset, state in enumerate(states):
            if state.status != StageState.READY or state.group_id != expected.group_id:
                raise RuntimeError(
                    f"replay ticket {expected.ticket_id} state {state.state_id} is not "
                    f"READY at group {expected.group_id}"
                )
            if state.request_id != expected.request_ids[offset]:
                raise RuntimeError("replay request identity mismatch")
            if state.chunk_index != expected.chunk_indices[offset] or (
                state.token_start,
                state.token_end,
            ) != expected.token_spans[offset]:
                raise RuntimeError("replay chunk identity mismatch")
            if state.state_version + 1 != expected.state_versions[offset]:
                raise RuntimeError(
                    f"replay state version mismatch for {state.state_id}: "
                    f"ready={state.state_version} expected_reserved="
                    f"{expected.state_versions[offset]}"
                )

        versions = queues.reserve(expected.state_ids, expected.ticket_id)
        if versions != expected.state_versions:
            raise RuntimeError("replay reservation versions diverged from the trace")
        if queues.queue_epoch != expected.queue_epoch:
            # Revert without changing the externally visible READY ordering.
            for state in states:
                state.status = StageState.READY
                state.reserved_ticket_id = None
                state.state_version -= 1
                queues._append_ready_id(state)
            queues.queue_epoch -= 1
            raise RuntimeError(
                f"replay queue epoch mismatch: {queues.queue_epoch + 1} != "
                f"{expected.queue_epoch}"
            )
        self.index += 1
        return expected

    def assert_exhausted(self) -> None:
        if not self.exhausted:
            raise RuntimeError(
                f"action replay stopped after {self.index}/{len(self.tickets)} tickets"
            )
