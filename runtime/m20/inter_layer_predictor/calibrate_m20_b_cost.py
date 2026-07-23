#!/usr/bin/env python3
"""Build an action-level M20-B cost calibration from completed policy runs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


KT_TIME_RE = re.compile(
    r"\[kt-time\]\s+layer=(?P<layer>-?\d+)\s+step=(?P<step>\d+)\s+"
    r"total=(?P<total>[0-9.]+)ms\s+submit=(?P<submit>[0-9.]+)\s+"
    r"mask=(?P<mask>[0-9.]+)\s+gpu=(?P<gpu>[0-9.]+)\s+"
    r"sync=(?P<sync>[0-9.]+)\s+merge=(?P<merge>[0-9.]+)\s+"
    r"cpu_wait=(?P<cpu_wait>[0-9.]+)ms\s+num_tokens=(?P<num_tokens>\d+)"
    r"(?:\s+cpu_entries=(?P<cpu_entries>\d+)\s+gpu_entries=(?P<gpu_entries>\d+)"
    r"\s+cpu_unique=(?P<cpu_unique>\d+)\s+gpu_unique=(?P<gpu_unique>\d+))?"
)
DISPATCH_RE = re.compile(
    r"\[kt-stage\]\s+action_dispatched\s+ticket=(?P<ticket>\d+).*?"
    r"group=(?P<group>\d+)\s+tokens=(?P<tokens>\d+)\s+"
    r"layers=\[(?P<start>\d+),(?P<end>\d+)\)"
)
COMPLETE_RE = re.compile(
    r"\[kt-stage\]\s+action_completed\s+ticket=(?P<ticket>\d+)"
)
TIMING_FIELDS = (
    "total",
    "submit",
    "mask",
    "gpu",
    "sync",
    "merge",
    "cpu_wait",
    "cpu_entries",
    "gpu_entries",
    "cpu_unique",
    "gpu_unique",
)


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def batch_request_id(
    prompt_index: int, chunk_index: int, token_start: int, token_end: int
) -> str:
    payload = [
        {
            "identity": ["prompt_global_index", str(int(prompt_index))],
            "token_start": int(token_start),
            "token_end": int(token_end),
            "chunk_index": int(chunk_index),
        }
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "batch:" + hashlib.sha256(encoded).hexdigest()


class OracleCoverageIndex:
    """Recompute actual route coverage for every frozen action plan."""

    def __init__(
        self,
        routes: Mapping[int, np.ndarray],
        *,
        prompt_indices: Sequence[int],
        route_top_k: int,
        measured_prompt_indices: Sequence[int] | None = None,
    ):
        self.routes = {int(key): value for key, value in routes.items()}
        self.prompt_indices = [int(value) for value in prompt_indices]
        self.measured_prompt_indices = {
            int(value)
            for value in (
                self.prompt_indices
                if measured_prompt_indices is None
                else measured_prompt_indices
            )
        }
        self.route_top_k = int(route_top_k)
        self._request_cache: dict[tuple[str, int, int, int], int] = {}

    @staticmethod
    def _trace_metadata(trace_path: Path) -> Mapping[str, Any]:
        for name in ("activation_stats.metadata.json", "summary.json"):
            metadata_path = trace_path.parent / name
            if not metadata_path.exists():
                continue
            value = read_json(metadata_path)
            if isinstance(value, Mapping):
                return value
        return {}

    @classmethod
    def from_provenance(
        cls, provenance: Mapping[str, Any], *, route_top_k: int
    ) -> "OracleCoverageIndex":
        args = provenance.get("args") or {}
        trace_path = Path(str(args.get("oracle_trace", ""))).resolve()
        if not trace_path.exists():
            raise FileNotFoundError(f"oracle trace is missing: {trace_path}")
        prompt_offset = int(args.get("prompt_offset", 0))
        num_prompts = int(args.get("num_prompts", 0))
        if num_prompts <= 0:
            raise ValueError("provenance has no measured prompts")
        measured_prompt_indices = list(
            range(prompt_offset, prompt_offset + num_prompts)
        )
        warmup_count = int(args.get("warmup_num_prompts", 0))
        warmup_offset = int(args.get("warmup_prompt_offset", 0))
        warmup_prompt_indices = list(
            range(warmup_offset, warmup_offset + warmup_count)
        )
        if set(measured_prompt_indices) & set(warmup_prompt_indices):
            raise ValueError("measured and warmup prompt ranges overlap")
        prompt_indices = sorted(
            {*measured_prompt_indices, *warmup_prompt_indices}
        )
        wanted = set(prompt_indices)
        metadata = cls._trace_metadata(trace_path)
        trace_prompt_offset = int(metadata.get("prompt_offset", 0))
        routes: dict[int, np.ndarray] = {}
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("ok", True):
                    continue
                request_id = int(row.get("request_id", 0))
                raw_prompt_index = row.get("prompt_global_index")
                if raw_prompt_index is None:
                    raw_prompt_index = row.get("prompt_index")
                if raw_prompt_index is None:
                    raw_prompt_index = trace_prompt_offset + request_id
                prompt_index = int(raw_prompt_index)
                if prompt_index not in wanted:
                    continue
                shape = row.get("routed_shape")
                encoded = row.get("routed_experts_base64")
                if not isinstance(shape, list) or len(shape) != 3 or not encoded:
                    raise ValueError(
                        f"oracle row {request_id} has no valid routed tensor"
                    )
                if int(shape[2]) != route_top_k:
                    raise ValueError(
                        f"oracle route top-k {shape[2]} != expected {route_top_k}"
                    )
                flat = np.frombuffer(
                    base64.b64decode(str(encoded).encode("utf-8")),
                    dtype=np.int32,
                )
                expected = math.prod(int(value) for value in shape)
                if int(flat.size) != expected:
                    raise ValueError(
                        f"oracle row {request_id} shape mismatch: "
                        f"{flat.size} != {expected}"
                    )
                routes[prompt_index] = flat.reshape(
                    int(shape[0]), int(shape[1]), int(shape[2])
                )
        missing = sorted(wanted - set(routes))
        if missing:
            raise ValueError(f"oracle trace is missing measured prompts: {missing}")
        return cls(
            routes,
            prompt_indices=prompt_indices,
            route_top_k=route_top_k,
            measured_prompt_indices=measured_prompt_indices,
        )

    def _resolve_prompt(
        self, request_id: str, chunk_index: int, token_start: int, token_end: int
    ) -> int:
        key = (
            str(request_id),
            int(chunk_index),
            int(token_start),
            int(token_end),
        )
        cached = self._request_cache.get(key)
        if cached is not None:
            return cached
        matches = [
            prompt_index
            for prompt_index in self.prompt_indices
            if batch_request_id(
                prompt_index, chunk_index, token_start, token_end
            )
            == request_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot uniquely resolve action request {request_id}: {matches}"
            )
        self._request_cache[key] = matches[0]
        return matches[0]

    def ticket_prompt_indices(self, ticket: Mapping[str, Any]) -> list[int]:
        request_ids = [str(value) for value in ticket.get("request_ids") or []]
        chunk_indices = [int(value) for value in ticket.get("chunk_indices") or []]
        token_spans = [
            tuple(int(item) for item in value)
            for value in ticket.get("token_spans") or []
        ]
        if not request_ids or not (
            len(request_ids) == len(chunk_indices) == len(token_spans)
        ):
            raise ValueError("ticket request/chunk/token-span metadata is incomplete")
        return [
            self._resolve_prompt(
                request_id, chunk_index, token_start, token_end
            )
            for request_id, chunk_index, (token_start, token_end) in zip(
                request_ids, chunk_indices, token_spans
            )
        ]

    def covered_routes(self, ticket: Mapping[str, Any]) -> float:
        token_spans = [
            tuple(int(item) for item in value)
            for value in ticket.get("token_spans") or []
        ]
        prompt_indices = self.ticket_prompt_indices(ticket)
        plans = {
            int(layer): {int(expert) for expert in experts}
            for layer, experts in ticket.get("layer_plans") or []
        }
        if not plans:
            raise ValueError("ticket has no layer plans")
        expected_entries = int(ticket["token_count"]) * len(plans) * self.route_top_k
        observed_entries = 0
        covered = 0
        for prompt_index, (token_start, token_end) in zip(
            prompt_indices, token_spans
        ):
            routed = self.routes[prompt_index]
            if not 0 <= token_start < token_end <= int(routed.shape[0]):
                raise ValueError(
                    f"token span [{token_start}, {token_end}) is outside oracle "
                    f"prompt {prompt_index} shape {routed.shape}"
                )
            for layer, experts in plans.items():
                if not 0 <= layer < int(routed.shape[1]):
                    raise ValueError(
                        f"layer {layer} is outside oracle shape {routed.shape}"
                    )
                values = routed[token_start:token_end, layer, :]
                if np.any(values < 0):
                    raise ValueError("oracle coverage contains invalid expert IDs")
                observed_entries += int(values.size)
                covered += sum(
                    int(np.count_nonzero(values == expert)) for expert in experts
                )
        if observed_entries != expected_entries:
            raise ValueError(
                f"oracle route entries {observed_entries} != expected {expected_entries}"
            )
        return float(covered)


def load_planned_tickets(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    tickets: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("record_type") == "header":
            header = dict(record.get("metadata") or {})
        elif record.get("record_type") == "action" and record.get("event") == "planned":
            tickets.append(dict(record["ticket"]))
    if header is None:
        raise ValueError(f"action trace has no header: {path}")
    if not tickets:
        raise ValueError(f"action trace has no planned tickets: {path}")
    ticket_ids = [int(ticket["ticket_id"]) for ticket in tickets]
    if ticket_ids != list(range(len(tickets))):
        raise ValueError(f"planned ticket IDs are not contiguous: {ticket_ids}")
    return header, tickets


def parse_action_timing(path: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    actions: dict[int, dict[str, Any]] = {}
    issues: list[str] = []
    current: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            dispatch = DISPATCH_RE.search(line)
            if dispatch:
                if current is not None:
                    raise ValueError(
                        f"nested action dispatch at {path}:{line_number}"
                    )
                current = {
                    "ticket_id": int(dispatch.group("ticket")),
                    "group_id": int(dispatch.group("group")),
                    "token_count": int(dispatch.group("tokens")),
                    "layer_start": int(dispatch.group("start")),
                    "layer_end": int(dispatch.group("end")),
                    "timing_rows": [],
                }
                continue
            timing = KT_TIME_RE.search(line)
            if timing and current is not None:
                row: dict[str, Any] = {}
                for key, value in timing.groupdict().items():
                    if value is None:
                        row[key] = 0
                    elif key in {
                        "layer",
                        "step",
                        "num_tokens",
                        "cpu_entries",
                        "gpu_entries",
                        "cpu_unique",
                        "gpu_unique",
                    }:
                        row[key] = int(value)
                    else:
                        row[key] = float(value)
                current["timing_rows"].append(row)
                continue
            completed = COMPLETE_RE.search(line)
            if not completed:
                continue
            ticket_id = int(completed.group("ticket"))
            if current is None or int(current["ticket_id"]) != ticket_id:
                raise ValueError(
                    f"completion without matching dispatch at {path}:{line_number}"
                )
            expected_layers = list(
                range(int(current["layer_start"]), int(current["layer_end"]))
            )
            observed_layers = [
                int(row["layer"]) for row in current["timing_rows"]
            ]
            complete = observed_layers == expected_layers
            if not complete:
                issues.append(
                    f"ticket {ticket_id}: timing layers {observed_layers} != "
                    f"{expected_layers}"
                )
            for field in TIMING_FIELDS:
                current[f"{field}_ms" if field not in {
                    "cpu_entries",
                    "gpu_entries",
                    "cpu_unique",
                    "gpu_unique",
                } else field] = sum(
                    float(row[field]) for row in current["timing_rows"]
                )
            current["first_group_visit"] = any(
                int(row["step"]) == 1 for row in current["timing_rows"]
            )
            current["timing_complete"] = complete
            actions[ticket_id] = current
            current = None
    if current is not None:
        raise ValueError(f"unterminated action dispatch in {path}")
    return actions, issues


def action_transport_rows(
    action_metrics: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    result: dict[int, dict[str, Any]] = {}
    issues: list[str] = []
    previous = {"host_prepare_ms": 0.0, "h2d_ms": 0.0}
    for index, metrics in enumerate(action_metrics):
        materialization = dict(metrics.get("action_materialization") or {})
        if not materialization:
            issues.append(f"action metric {index} has no materialization")
            continue
        ticket_id = int(materialization["ticket_id"])
        current = {
            "host_prepare_ms": float(metrics.get("host_prepare_ms", 0.0)),
            "h2d_ms": float(metrics.get("h2d_ms", 0.0)),
        }
        host_delta = current["host_prepare_ms"] - previous["host_prepare_ms"]
        copy_delta = current["h2d_ms"] - previous["h2d_ms"]
        if host_delta < -1e-6 or copy_delta < -1e-6:
            issues.append(
                f"ticket {ticket_id}: cumulative transport counters decreased"
            )
        result[ticket_id] = {
            "materialization": str(materialization.get("materialization", "")),
            "h2d_experts": int(materialization.get("h2d_experts", 0)),
            "d2d_experts": int(materialization.get("d2d_experts", 0)),
            "h2d_bytes": int(materialization.get("h2d_bytes", 0)),
            "d2d_bytes": int(materialization.get("d2d_bytes", 0)),
            "host_prepare_ms": max(0.0, host_delta),
            "copy_event_ms": max(0.0, copy_delta),
        }
        previous = current
    return result, issues


def infer_route_top_k(provenance: Mapping[str, Any], override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("route top-k must be positive")
        return int(override)
    trace_path = Path(str((provenance.get("args") or {}).get("oracle_trace", "")))
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                shape = row.get("routed_shape")
                if isinstance(shape, list) and len(shape) == 3:
                    return int(shape[2])
    raise ValueError("cannot infer route top-k; pass --route-top-k")


def legacy_covered_routes(
    ticket: Mapping[str, Any], trace_header: Mapping[str, Any]
) -> tuple[float, str]:
    score = dict(ticket.get("score") or {})
    if "covered_route_entries" in score:
        return float(score["covered_route_entries"]), "explicit"
    route_gain = float(
        ((trace_header.get("cost_model") or {}).get("route_entry_gain_ms", 0.0))
    )
    if route_gain <= 0:
        raise ValueError(
            "legacy trace has neither covered_route_entries nor positive route gain"
        )
    return float(score.get("compute_gain_ms", 0.0)) / route_gain, "legacy_ratio"


def resolve_run_dir(experiment_dir: Path, row: Mapping[str, Any]) -> Path:
    local = experiment_dir / str(row["run_name"])
    if local.exists():
        return local
    result_dir = Path(str(row.get("result_dir", "")))
    if result_dir.exists():
        return result_dir
    raise FileNotFoundError(f"run directory is missing for {row['run_name']}")


def build_action_rows(
    experiment_dir: Path,
    *,
    route_top_k: int,
    min_token_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(experiment_dir / "summary.json")
    provenance = read_json(experiment_dir / "provenance.json")
    coverage_index = OracleCoverageIndex.from_provenance(
        provenance, route_top_k=route_top_k
    )
    planners = {
        (
            int(row["repeat"]),
            str(row.get("policy_label", row.get("stage_policy", "fifo"))),
        ): row
        for row in summary
        if row.get("materialization") == "full" and row.get("status") == "ok"
    }
    delta_rows = [
        row
        for row in summary
        if row.get("materialization") == "delta" and row.get("status") == "ok"
    ]
    if not delta_rows:
        raise ValueError("experiment has no successful delta rows")

    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    input_hashes: dict[str, str] = {}
    for delta in delta_rows:
        repeat = int(delta["repeat"])
        policy = str(
            delta.get("policy_label", delta.get("stage_policy", "fifo"))
        )
        planner = planners.get((repeat, policy))
        if planner is None:
            raise ValueError(f"missing full planner for repeat={repeat}, policy={policy}")
        planner_dir = resolve_run_dir(experiment_dir, planner)
        delta_dir = resolve_run_dir(experiment_dir, delta)
        trace_path = planner_dir / "actions.jsonl"
        log_path = delta_dir / "server.log"
        status_path = delta_dir / "runner_status.json"
        input_hashes[str(trace_path)] = sha256(trace_path)
        input_hashes[str(log_path)] = sha256(log_path)
        input_hashes[str(status_path)] = sha256(status_path)

        header, tickets = load_planned_tickets(trace_path)
        timing, timing_issues = parse_action_timing(log_path)
        status = read_json(status_path)
        transport, transport_issues = action_transport_rows(
            ((status.get("group_profile") or {}).get("action_metrics") or [])
        )
        issues.extend(
            f"{policy}/r{repeat}: {value}"
            for value in [*timing_issues, *transport_issues]
        )
        if len(tickets) != len(timing) or len(tickets) != len(transport):
            issues.append(
                f"{policy}/r{repeat}: ticket/timing/transport counts are "
                f"{len(tickets)}/{len(timing)}/{len(transport)}"
            )

        for ticket in tickets:
            ticket_id = int(ticket["ticket_id"])
            timing_row = timing.get(ticket_id)
            transport_row = transport.get(ticket_id)
            if timing_row is None or transport_row is None:
                continue
            configured_covered, configured_covered_source = legacy_covered_routes(
                ticket, header
            )
            ticket_prompt_indices = coverage_index.ticket_prompt_indices(ticket)
            measured_action = all(
                value in coverage_index.measured_prompt_indices
                for value in ticket_prompt_indices
            )
            covered = coverage_index.covered_routes(ticket)
            covered_source = "oracle_recomputed"
            group_layers = len(ticket.get("layer_plans") or [])
            token_count = int(ticket["token_count"])
            total_routes = float(token_count * group_layers * route_top_k)
            if covered < -1e-6 or covered > total_routes + 1e-6:
                issues.append(
                    f"{policy}/r{repeat}/ticket{ticket_id}: covered routes "
                    f"{covered} outside [0, {total_routes}]"
                )
            row = {
                "repeat": repeat,
                "policy": policy,
                "ticket_id": ticket_id,
                "group_id": int(ticket["group_id"]),
                "state_ids": [int(value) for value in ticket["state_ids"]],
                "prompt_indices": ticket_prompt_indices,
                "measured_action": measured_action,
                "cohort_size": len(ticket["state_ids"]),
                "token_count": token_count,
                "group_layers": group_layers,
                "route_top_k": route_top_k,
                "total_route_entries": total_routes,
                "covered_route_entries": covered,
                "covered_route_source": covered_source,
                "configured_covered_route_entries": configured_covered,
                "configured_covered_route_source": configured_covered_source,
                "uncovered_route_entries": max(0.0, total_routes - covered),
                "coverage_fraction": covered / total_routes if total_routes else 0.0,
                "configured_compute_gain_ms": float(
                    (ticket.get("score") or {}).get("compute_gain_ms", 0.0)
                ),
                "configured_materialization_ms": float(
                    (ticket.get("score") or {}).get("materialization_ms", 0.0)
                ),
                "fallback": str(ticket.get("fallback", "")),
                **{
                    key: value
                    for key, value in timing_row.items()
                    if key != "timing_rows"
                },
                **transport_row,
            }
            row["observed_materialization_ms"] = (
                row["host_prepare_ms"] + row["copy_event_ms"]
            )
            row["primary_calibration_row"] = bool(
                row["timing_complete"]
                and not row["first_group_visit"]
                and row["measured_action"]
                and token_count >= min_token_count
            )
            rows.append(row)

    metadata = {
        "issues": issues,
        "input_hashes": input_hashes,
        "provenance": provenance,
        "policies": sorted({str(row["policy"]) for row in rows}),
        "repeats": sorted({int(row["repeat"]) for row in rows}),
    }
    return rows, metadata


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    errors = predicted - actual
    absolute = np.abs(errors)
    mean_actual = float(np.mean(actual)) if actual.size else 0.0
    denominator = float(np.sum((actual - mean_actual) ** 2))
    r2 = (
        1.0 - float(np.sum(errors**2)) / denominator
        if actual.size > 1 and denominator > 0
        else None
    )
    mae = float(np.mean(absolute)) if actual.size else None
    return {
        "n": int(actual.size),
        "mean_actual_ms": mean_actual if actual.size else None,
        "mae_ms": mae,
        "normalized_mae": (
            mae / mean_actual if mae is not None and mean_actual > 0 else None
        ),
        "rmse_ms": (
            float(math.sqrt(float(np.mean(errors**2)))) if actual.size else None
        ),
        "p95_absolute_error_ms": (
            percentile(absolute.tolist(), 0.95) if actual.size else None
        ),
        "r2": r2,
    }


def compute_design(
    rows: Sequence[Mapping[str, Any]],
    *,
    groups: Sequence[int],
    token_scale: float,
    route_scale: float,
) -> np.ndarray:
    values = []
    for row in rows:
        group_id = int(row["group_id"])
        values.append(
            [float(group_id == group) for group in groups]
            + [
                float(row["token_count"]) / token_scale,
                float(row["uncovered_route_entries"]) / route_scale,
            ]
        )
    return np.asarray(values, dtype=np.float64)


def fit_compute_model(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    token_scale: float,
    route_scale: float,
    groups: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not train_rows or not test_rows:
        raise ValueError("compute calibration requires non-empty train and test rows")
    model_groups = (
        sorted({int(row["group_id"]) for row in train_rows})
        if groups is None
        else [int(value) for value in groups]
    )
    if any(int(row["group_id"]) not in model_groups for row in test_rows):
        raise ValueError("test set contains a group absent from the compute model")
    x_train = compute_design(
        train_rows,
        groups=model_groups,
        token_scale=token_scale,
        route_scale=route_scale,
    )
    y_train = np.asarray([float(row[target]) for row in train_rows])
    coefficients, _residuals, rank, singular = np.linalg.lstsq(
        x_train, y_train, rcond=None
    )
    x_test = compute_design(
        test_rows,
        groups=model_groups,
        token_scale=token_scale,
        route_scale=route_scale,
    )
    y_test = np.asarray([float(row[target]) for row in test_rows])
    predicted = x_test @ coefficients
    route_coefficient = float(coefficients[-1]) / route_scale
    return {
        "target": target,
        "groups": model_groups,
        "feature_names": [
            *[f"group_{group}_intercept" for group in model_groups],
            f"token_count/{token_scale:g}",
            f"uncovered_route_entries/{route_scale:g}",
        ],
        "coefficients": [float(value) for value in coefficients],
        "route_entry_gain_ms": route_coefficient,
        "rank": int(rank),
        "condition_number": (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0
            else None
        ),
        "train": regression_metrics(y_train, x_train @ coefficients),
        "test": regression_metrics(y_test, predicted),
    }


def fit_transport_model(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not train_rows or not test_rows:
        raise ValueError("transport calibration requires non-empty train and test rows")

    def fit_target(target: str, features: Sequence[str]) -> dict[str, Any]:
        x_train = np.asarray(
            [[float(row[key]) for key in features] for row in train_rows],
            dtype=np.float64,
        )
        y_train = np.asarray([float(row[target]) for row in train_rows])
        coefficients, _residuals, rank, singular = np.linalg.lstsq(
            x_train, y_train, rcond=None
        )
        x_test = np.asarray(
            [[float(row[key]) for key in features] for row in test_rows],
            dtype=np.float64,
        )
        y_test = np.asarray([float(row[target]) for row in test_rows])
        return {
            "target": target,
            "features": list(features),
            "coefficients": [float(value) for value in coefficients],
            "rank": int(rank),
            "condition_number": (
                float(singular[0] / singular[-1])
                if singular.size and singular[-1] > 0
                else None
            ),
            "train": regression_metrics(y_train, x_train @ coefficients),
            "test": regression_metrics(y_test, x_test @ coefficients),
        }

    prepare = fit_target("host_prepare_ms", ["h2d_experts"])
    copy = fit_target("copy_event_ms", ["h2d_experts", "d2d_experts"])
    h2d_effective = prepare["coefficients"][0] + copy["coefficients"][0]
    return {
        "host_prepare": prepare,
        "copy_event": copy,
        "h2d_effective_ms_per_expert": h2d_effective,
        "d2d_effective_ms_per_expert": copy["coefficients"][1],
    }


def holdout_fits(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    fit: Any,
) -> list[dict[str, Any]]:
    values = sorted({row[key] for row in rows}, key=str)
    if len(values) < 2:
        return []
    result = []
    for value in values:
        train = [row for row in rows if row[key] != value]
        test = [row for row in rows if row[key] == value]
        model = fit(train, test)
        result.append({"holdout": value, **model})
    return result


def calibration_gate(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    compute: Mapping[str, Any],
    transport: Mapping[str, Any],
    *,
    max_normalized_mae: float,
) -> dict[str, Any]:
    provenance_args = (metadata.get("provenance") or {}).get("args") or {}
    reasons: list[str] = []
    if int(provenance_args.get("seq_len", 0)) < 2048:
        reasons.append("sequence length is below 2048")
    if int(provenance_args.get("num_prompts", 0)) < 8:
        reasons.append("fewer than 8 measured prompts")
    if len(metadata.get("repeats") or []) < 3:
        reasons.append("fewer than 3 independent repeats")
    required_policies = {"fifo", "min_delta", "cost_oracle", "frequency"}
    observed_policies = set(metadata.get("policies") or [])
    if not required_policies.issubset(observed_policies):
        missing = sorted(required_policies - observed_policies)
        reasons.append(
            "required policy coverage is incomplete: " + ", ".join(missing)
        )
    if metadata.get("issues"):
        reasons.append("input alignment issues are present")
    coverage_sources = {str(row.get("covered_route_source")) for row in rows}
    if coverage_sources != {"oracle_recomputed"}:
        reasons.append(
            "route coverage was not fully recomputed from the oracle trace"
        )
    primary_count = sum(bool(row["primary_calibration_row"]) for row in rows)
    if primary_count < 120:
        reasons.append("fewer than 120 steady full-chunk action rows")

    required_targets = ("cpu_wait_ms", "total_ms")
    for holdout_name in ("policy", "repeat"):
        holdouts = compute.get(f"{holdout_name}_holdouts") or {}
        for target in required_targets:
            models = holdouts.get(target) or []
            if not models:
                reasons.append(
                    f"{target} has no {holdout_name} holdout validation"
                )
                continue
            for model in models:
                if float(model["route_entry_gain_ms"]) <= 0:
                    reasons.append(
                        f"{target} route coefficient is non-positive for "
                        f"{holdout_name} holdout {model['holdout']}"
                    )
                error = model["test"].get("normalized_mae")
                if error is None or float(error) > max_normalized_mae:
                    reasons.append(
                        f"{target} normalized MAE exceeds "
                        f"{max_normalized_mae:.2f} for {holdout_name} "
                        f"holdout {model['holdout']}"
                    )

    h2d = float(transport["all_rows"]["h2d_effective_ms_per_expert"])
    d2d = float(transport["all_rows"]["d2d_effective_ms_per_expert"])
    if h2d <= 0 or d2d < 0:
        reasons.append("transport coefficients are not physically valid")
    return {
        "authorized": not reasons,
        "max_normalized_mae": max_normalized_mae,
        "primary_action_rows": primary_count,
        "reasons": reasons,
    }


def fmt(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    gate = result["gate"]
    compute = result["compute"]
    transport = result["transport"]
    lines = [
        "# M20-B retrospective cost calibration",
        "",
        f"Generated: {result['generated_at']}",
        "",
        f"Runtime values authorized: `{gate['authorized']}`.",
        "",
        "## Dataset",
        "",
        f"- actions: {result['dataset']['actions']}",
        f"- primary steady full-chunk actions: {gate['primary_action_rows']}",
        f"- policies: {', '.join(result['dataset']['policies'])}",
        f"- repeats: {result['dataset']['repeats']}",
        f"- route top-k: {result['dataset']['route_top_k']}",
        "",
        "## Compute model",
        "",
        "| Target | Route gain ms/entry | Train NMAE | Fit NMAE |",
        "|---|---:|---:|---:|",
    ]
    for target, model in compute["all_rows"].items():
        lines.append(
            f"| {target} | {fmt(model['route_entry_gain_ms'], 6)} | "
            f"{fmt(model['train']['normalized_mae'])} | "
            f"{fmt(model['test']['normalized_mae'])} |"
        )
    lines += [
        "",
        "The coefficient is the marginal action-level effect of one additional "
        "uncovered oracle route entry after controlling for group and token count. "
        "Policy holdouts, not the all-row fit, determine the error gate.",
        "",
        "## Transport model",
        "",
        f"- effective H2D: {fmt(transport['all_rows']['h2d_effective_ms_per_expert'])} ms/expert",
        f"- effective D2D: {fmt(transport['all_rows']['d2d_effective_ms_per_expert'])} ms/expert",
        "",
        "Effective H2D is host preparation plus copy-event time. D2D is inferred "
        "from mixed delta actions and remains weaker evidence than a dedicated microbenchmark.",
        "",
        "## Gate",
        "",
    ]
    if gate["reasons"]:
        lines.extend(f"- {reason}" for reason in gate["reasons"])
    else:
        lines.append("- all calibration gates passed")
    lines += [
        "",
        "The script never edits runtime parameters. Use candidate values only when "
        "`authorized=true`; otherwise run a dedicated long-context, repeated calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-top-k", type=int)
    parser.add_argument("--min-token-count", type=int)
    parser.add_argument("--max-normalized-mae", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.max_normalized_mae < 1:
        parser.error("--max-normalized-mae must be in (0, 1)")
    args.experiment_dir = args.experiment_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> int:
    args = parse_args()
    provenance = read_json(args.experiment_dir / "provenance.json")
    route_top_k = infer_route_top_k(provenance, args.route_top_k)
    chunk_size = int(
        args.min_token_count
        if args.min_token_count is not None
        else (provenance.get("args") or {}).get("chunked_prefill_size", 256)
    )
    if chunk_size <= 0:
        raise ValueError("minimum token count must be positive")
    rows, metadata = build_action_rows(
        args.experiment_dir,
        route_top_k=route_top_k,
        min_token_count=chunk_size,
    )
    primary = [row for row in rows if row["primary_calibration_row"]]
    if not primary:
        raise ValueError("no steady full-chunk actions are available for calibration")
    groups = sorted({int(row["group_id"]) for row in primary})
    route_scale = float(chunk_size * route_top_k * statistics.median(
        int(row["group_layers"]) for row in primary
    ))
    token_scale = float(chunk_size)

    def compute_fit(target: str, train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return fit_compute_model(
            train,
            test,
            target=target,
            token_scale=token_scale,
            route_scale=route_scale,
            groups=groups,
        )

    compute_all = {
        target: compute_fit(target, primary, primary)
        for target in ("cpu_wait_ms", "total_ms")
    }
    policy_holdouts = {
        target: holdout_fits(
            primary,
            key="policy",
            fit=lambda train, test, target=target: compute_fit(
                target, train, test
            ),
        )
        for target in ("cpu_wait_ms", "total_ms")
    }
    repeat_holdouts = {
        target: holdout_fits(
            primary,
            key="repeat",
            fit=lambda train, test, target=target: compute_fit(
                target, train, test
            ),
        )
        for target in ("cpu_wait_ms", "total_ms")
    }
    transport_all = fit_transport_model(rows, rows)
    transport_policy = holdout_fits(
        rows,
        key="policy",
        fit=fit_transport_model,
    )
    compute = {
        "feature_contract": {
            "group_fixed_effects": groups,
            "token_scale": token_scale,
            "route_scale": route_scale,
            "filters": ["timing_complete", "not first_group_visit", f"token_count >= {chunk_size}"],
        },
        "all_rows": compute_all,
        "policy_holdouts": policy_holdouts,
        "repeat_holdouts": repeat_holdouts,
    }
    transport = {
        "all_rows": transport_all,
        "policy_holdouts": transport_policy,
    }
    gate = calibration_gate(
        rows,
        metadata,
        compute,
        transport,
        max_normalized_mae=args.max_normalized_mae,
    )
    candidate = {
        "authorized": gate["authorized"],
        "route_entry_gain_ms": (
            compute_all["total_ms"]["route_entry_gain_ms"]
            if gate["authorized"]
            else None
        ),
        "h2d_expert_ms": (
            transport_all["h2d_effective_ms_per_expert"]
            if gate["authorized"]
            else None
        ),
        "d2d_expert_ms": (
            transport_all["d2d_effective_ms_per_expert"]
            if gate["authorized"]
            else None
        ),
        "diagnostic_only": {
            "route_entry_gain_ms": compute_all["total_ms"]["route_entry_gain_ms"],
            "h2d_expert_ms": transport_all["h2d_effective_ms_per_expert"],
            "d2d_expert_ms": transport_all["d2d_effective_ms_per_expert"],
        },
    }
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibrator": str(Path(__file__).resolve()),
        "calibrator_sha256": sha256(Path(__file__).resolve()),
        "experiment_dir": str(args.experiment_dir),
        "dataset": {
            "actions": len(rows),
            "primary_actions": len(primary),
            "policies": metadata["policies"],
            "repeats": metadata["repeats"],
            "route_top_k": route_top_k,
            "min_token_count": chunk_size,
            "covered_route_sources": dict(
                sorted(
                    {
                        source: sum(row["covered_route_source"] == source for row in rows)
                        for source in {row["covered_route_source"] for row in rows}
                    }.items()
                )
            ),
        },
        "input_hashes": metadata["input_hashes"],
        "issues": metadata["issues"],
        "compute": compute,
        "transport": transport,
        "gate": gate,
        "runtime_candidate": candidate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "calibration.json", result)
    with (args.output_dir / "action_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_report(args.output_dir / "REPORT.md", result)
    print(json.dumps({"output_dir": str(args.output_dir), "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
