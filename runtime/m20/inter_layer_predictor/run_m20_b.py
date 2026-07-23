#!/usr/bin/env python3
"""Run paired M20-B stage planner and exact action-replay experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_m20_a0_a1 as common
from sglang.srt.managers.kt_stage_scheduler import ActionTrace


RUNNER = Path(__file__).resolve()
DEFAULT_EXCLUSIVE_LOCK = Path("/tmp/m20_b_formal_exclusive.lock")


def cpu_affinity_snapshot() -> dict[str, Any]:
    logical_cpus = sorted(os.sched_getaffinity(0))
    physical_ids: list[tuple[int, int]] = []
    for cpu in logical_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package_id = int((topology / "physical_package_id").read_text().strip())
        core_id = int((topology / "core_id").read_text().strip())
        physical_ids.append((package_id, core_id))
    unique_physical_ids = sorted(set(physical_ids))
    return {
        "logical_cpus": logical_cpus,
        "logical_cpu_count": len(logical_cpus),
        "physical_core_count": len(unique_physical_ids),
        "physical_core_ids": [list(value) for value in unique_physical_ids],
        "has_smt_siblings": len(logical_cpus) != len(unique_physical_ids),
    }


def acquire_exclusive_lock(
    path: Path,
    snapshot: dict[str, Any],
    cpu_threads: int,
    *,
    check_competing_runners: bool = True,
):
    if snapshot["has_smt_siblings"]:
        raise RuntimeError(
            "exclusive mode requires an affinity with at most one logical CPU "
            "per physical core"
        )
    if int(snapshot["physical_core_count"]) < cpu_threads:
        raise RuntimeError(
            "exclusive mode CPU affinity has fewer physical cores than "
            f"--cpu-threads ({snapshot['physical_core_count']} < {cpu_threads})"
        )
    if check_competing_runners:
        competing_pids = []
        for proc_path in Path("/proc").glob("[0-9]*"):
            pid = int(proc_path.name)
            if pid == os.getpid():
                continue
            try:
                argv = (proc_path / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(
                Path(value.decode(errors="replace")).name == RUNNER.name
                for value in argv
                if value
            ):
                competing_pids.append(pid)
        if competing_pids:
            raise RuntimeError(
                "exclusive mode found other M20-B runners: "
                + ", ".join(str(value) for value in sorted(competing_pids))
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"another exclusive M20-B runner holds {path}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cpu_affinity": snapshot,
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def policy_order_for_repeat(
    policies: list[str], repeat: int, *, rotate: bool
) -> list[str]:
    if not rotate or len(policies) < 2:
        return list(policies)
    midpoint = (len(policies) + 1) // 2
    offsets = [0, *range(midpoint, len(policies)), *range(1, midpoint)]
    offset = offsets[(repeat - 1) % len(offsets)]
    return [*policies[offset:], *policies[:offset]]


def replay_modes_for_repeat(load_mode: str, repeat: int) -> list[str]:
    """Balance sync/async replay order without changing either frozen trace."""

    if load_mode != "both":
        return [load_mode]
    return ["async", "sync"] if repeat % 2 == 0 else ["sync", "async"]


def request_outputs(row: dict[str, Any]) -> dict[int, str]:
    path = Path(row["result_dir"]) / "default_request_metrics.jsonl"
    if not path.exists():
        return {}
    return {
        int(item["prompt_index"]): str(item.get("generated_text", ""))
        for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }


def load_trace_stats(path: Path, cohort_size: int) -> dict[str, Any]:
    trace = ActionTrace.load(path)
    planned_tickets = list(trace.iter_tickets("planned"))
    materialized_tickets = list(trace.iter_tickets("materialized"))
    tickets = materialized_tickets or planned_tickets
    event_counts = Counter(
        str(record.get("event"))
        for record in trace.records[1:]
        if record.get("record_type") == "action"
    )
    cohort_sizes = [len(ticket.state_ids) for ticket in tickets]
    copy_kinds = Counter(
        op.kind.value for ticket in tickets for op in ticket.copy_ops
    )
    planned_copy_kinds = Counter(
        op.kind.value for ticket in planned_tickets for op in ticket.copy_ops
    )
    fallback_counts = Counter(ticket.fallback or "none" for ticket in tickets)
    policy_counts = Counter(ticket.policy for ticket in tickets)
    net_gains = [ticket.score.net_gain_ms for ticket in tickets]
    return {
        "path": str(path),
        "file_sha256": common.sha256(path),
        "chain_hash": trace.trace_hash,
        "metadata": dict(trace.metadata),
        "records": len(trace.records),
        "planned_actions": len(planned_tickets),
        "materialized_actions": len(materialized_tickets),
        "materialization_complete": bool(planned_tickets)
        and len(materialized_tickets) == len(planned_tickets),
        "event_counts": dict(sorted(event_counts.items())),
        "cohort_size_histogram": dict(
            sorted(Counter(cohort_sizes).items())
        ),
        "max_observed_cohort": max(cohort_sizes, default=0),
        "shared_actions": sum(size > 1 for size in cohort_sizes),
        "partial_actions": sum(size < cohort_size for size in cohort_sizes),
        "tickets_with_layer_plans": sum(bool(ticket.layer_plans) for ticket in tickets),
        "all_tickets_have_layer_plans": bool(tickets)
        and all(bool(ticket.layer_plans) for ticket in tickets),
        "target_buffer_histogram": dict(
            sorted(
                Counter(
                    ticket.target_buffer_id
                    for ticket in tickets
                    if ticket.target_buffer_id is not None
                ).items()
            )
        ),
        "copy_kind_histogram": dict(sorted(copy_kinds.items())),
        "predicted_copy_kind_histogram": dict(sorted(planned_copy_kinds.items())),
        "predicted_zero_copy_actions": sum(
            not ticket.copy_ops for ticket in planned_tickets
        ),
        "predicted_h2d_bytes": sum(
            op.nbytes
            for ticket in planned_tickets
            for op in ticket.copy_ops
            if op.kind.value == "h2d"
        ),
        "predicted_d2d_bytes": sum(
            op.nbytes
            for ticket in planned_tickets
            for op in ticket.copy_ops
            if op.kind.value == "d2d"
        ),
        "policy_histogram": dict(sorted(policy_counts.items())),
        "fallback_histogram": dict(sorted(fallback_counts.items())),
        "positive_net_gain_actions": sum(value > 0 for value in net_gains),
        "net_gain_ms_sum": sum(net_gains),
        "net_gain_ms_min": min(net_gains, default=0.0),
        "net_gain_ms_max": max(net_gains, default=0.0),
        "h2d_bytes": sum(
            op.nbytes
            for ticket in tickets
            for op in ticket.copy_ops
            if op.kind.value == "h2d"
        ),
        "d2d_bytes": sum(
            op.nbytes
            for ticket in tickets
            for op in ticket.copy_ops
            if op.kind.value == "d2d"
        ),
    }


def action_count(row: dict[str, Any]) -> int:
    profile = row.get("group_profile") or {}
    counts = profile.get("event_counts") or {}
    return int(counts.get("action_end", 0))


def audit_replacement_budget(
    row: dict[str, Any], max_replacements: int
) -> dict[str, Any]:
    profile = row.get("group_profile") or {}
    action_metrics = profile.get("action_metrics") or []
    checked_logical_source_actions = 0
    checked_physical_source_actions = 0
    violations: list[dict[str, Any]] = []
    for index, metrics in enumerate(action_metrics):
        materialization = metrics.get("action_materialization") or {}
        if not materialization:
            continue
        changed_by_layer = {
            int(layer_idx): int(changed)
            for layer_idx, changed in (
                materialization.get("changed_by_layer") or {}
            ).items()
        }
        max_changed = max(changed_by_layer.values(), default=0)
        h2d_experts = int(materialization.get("h2d_experts", 0))
        changed_total = sum(changed_by_layer.values())
        reasons = []
        if materialization.get("logical_source_plan_present"):
            checked_logical_source_actions += 1
        if (
            materialization.get("logical_source_plan_present")
            and max_changed > max_replacements
        ):
            reasons.append(
                f"max_changed={max_changed} exceeds K={max_replacements}"
            )
        if (
            "physical_source_matches_logical" in materialization
        ):
            physical_source_matches_logical = bool(
                materialization["physical_source_matches_logical"]
            )
        else:
            # Artifacts produced before the exact-source field retain the
            # legacy audit behavior so existing reports remain reproducible.
            physical_source_matches_logical = (
                materialization.get("source_buffer_id") is not None
            )
        if (
            physical_source_matches_logical
            and materialization.get("materialization") == "delta"
        ):
            checked_physical_source_actions += 1
        if (
            physical_source_matches_logical
            and materialization.get("materialization") == "delta"
            and h2d_experts > changed_total
        ):
            reasons.append(
                f"h2d_experts={h2d_experts} exceeds changed={changed_total}"
            )
        if reasons:
            violations.append(
                {
                    "action_index": index,
                    "ticket_id": materialization.get("ticket_id"),
                    "group_id": materialization.get("group_id"),
                    "reasons": reasons,
                }
            )
    return {
        "passed": checked_logical_source_actions > 0 and not violations,
        "checked_logical_source_actions": checked_logical_source_actions,
        "checked_physical_source_actions": checked_physical_source_actions,
        "violations": violations,
    }


def compare_pair(
    planner: dict[str, Any],
    replay: dict[str, Any],
    trace_stats: dict[str, Any],
) -> dict[str, Any]:
    planner_outputs = request_outputs(planner) if planner.get("status") == "ok" else {}
    replay_outputs = request_outputs(replay) if replay.get("status") == "ok" else {}
    prompt_indices = sorted(set(planner_outputs) | set(replay_outputs))
    mismatches = [
        index
        for index in prompt_indices
        if planner_outputs.get(index) != replay_outputs.get(index)
    ]
    expected_actions = int(trace_stats["planned_actions"])
    planner_actions = action_count(planner)
    replay_actions = action_count(replay)
    return {
        "repeat": int(planner["repeat"]),
        "policy": str(
            planner.get("policy_label", planner.get("stage_policy", "fifo"))
        ),
        "planner": planner["run_name"],
        "replay": replay["run_name"],
        "replay_load_mode": str(replay.get("load_mode", "sync")),
        "outputs_match": bool(prompt_indices) and not mismatches,
        "mismatch_prompt_indices": mismatches,
        "prompt_count": len(prompt_indices),
        "expected_actions": expected_actions,
        "planner_actions": planner_actions,
        "replay_actions": replay_actions,
        "action_counts_match": (
            expected_actions > 0
            and planner_actions == expected_actions
            and replay_actions == expected_actions
        ),
        "trace_file_sha256": trace_stats["file_sha256"],
        "trace_chain_hash": trace_stats["chain_hash"],
    }


def action_materializations(row: dict[str, Any]) -> list[dict[str, Any]]:
    profile = row.get("group_profile") or {}
    return [
        dict(metrics.get("action_materialization") or {})
        for metrics in profile.get("action_metrics") or []
    ]


def final_action_metrics(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("group_profile") or {}
    return dict(profile.get("last_action_metrics") or {})


def nested_number(row: dict[str, Any], *keys: str) -> float | None:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if value is None:
        return None
    return float(value)


def compare_pipeline_replays(
    sync: dict[str, Any], async_row: dict[str, Any]
) -> dict[str, Any]:
    sync_outputs = request_outputs(sync) if sync.get("status") == "ok" else {}
    async_outputs = (
        request_outputs(async_row) if async_row.get("status") == "ok" else {}
    )
    prompt_indices = sorted(set(sync_outputs) | set(async_outputs))
    output_mismatches = [
        prompt_index
        for prompt_index in prompt_indices
        if sync_outputs.get(prompt_index) != async_outputs.get(prompt_index)
    ]
    sync_actions = action_materializations(sync)
    async_actions = action_materializations(async_row)
    action_count_max = max(len(sync_actions), len(async_actions))
    materialization_mismatches = [
        index
        for index in range(action_count_max)
        if (
            sync_actions[index] if index < len(sync_actions) else None
        )
        != (
            async_actions[index] if index < len(async_actions) else None
        )
    ]
    copy_op_mismatches = [
        index
        for index in range(action_count_max)
        if (
            (sync_actions[index].get("copy_ops") or [])
            if index < len(sync_actions)
            else None
        )
        != (
            (async_actions[index].get("copy_ops") or [])
            if index < len(async_actions)
            else None
        )
    ]
    sync_metrics = final_action_metrics(sync)
    async_metrics = final_action_metrics(async_row)
    expected_lookaheads = max(action_count(async_row) - 1, 0)
    pipeline_counters = {
        "expected_lookaheads": expected_lookaheads,
        "expected_nonblocking_action_ends": action_count(async_row),
        "submitted": int(async_metrics.get("pipeline_prefetch_submitted", 0)),
        "adopted": int(async_metrics.get("pipeline_prefetch_adopted", 0)),
        "nonblocking_action_ends": int(
            async_metrics.get("pipeline_end_action_nonblocking", 0)
        ),
        "failures": int(async_metrics.get("pipeline_prefetch_failures", 0)),
        "mismatches": int(async_metrics.get("pipeline_prefetch_mismatches", 0)),
        "ready_misses": int(async_metrics.get("ready_misses", 0)),
        "blocks": int(async_metrics.get("block_count", 0)),
        "active_overwrite_rejections": int(
            async_metrics.get("active_overwrite_rejections", 0)
        ),
    }
    pipeline_counters_passed = (
        pipeline_counters["submitted"] == expected_lookaheads
        and pipeline_counters["adopted"] == expected_lookaheads
        and pipeline_counters["nonblocking_action_ends"]
        == pipeline_counters["expected_nonblocking_action_ends"]
        and all(
            pipeline_counters[key] == 0
            for key in (
                "failures",
                "mismatches",
                "active_overwrite_rejections",
            )
        )
        # A lookahead can be valid and fully adopted while its CUDA event
        # still has a residual tail at the next action boundary.  With the
        # formal block miss policy every such ready miss must be accounted as
        # one block, but neither counter is a replay-correctness failure.
        and pipeline_counters["blocks"] == pipeline_counters["ready_misses"]
    )
    transport_fields = (
        "h2d_experts",
        "d2d_experts",
        "h2d_bytes",
        "d2d_bytes",
        "retained_experts",
        "retained_bytes",
    )
    transport_match = all(
        int(sync_metrics.get(key, 0)) == int(async_metrics.get(key, 0))
        for key in transport_fields
    )
    sync_throughput = nested_number(sync, "client", "prefill_tokens_per_s")
    async_throughput = nested_number(
        async_row, "client", "prefill_tokens_per_s"
    )
    sync_p50 = nested_number(sync, "client", "latency_p50_s")
    async_p50 = nested_number(async_row, "client", "latency_p50_s")
    return {
        "repeat": int(sync["repeat"]),
        "policy": str(sync.get("policy_label", sync.get("stage_policy", "fifo"))),
        "sync_run": sync["run_name"],
        "async_run": async_row["run_name"],
        "status_ok": sync.get("status") == "ok" and async_row.get("status") == "ok",
        "prompt_count": len(prompt_indices),
        "outputs_match": bool(prompt_indices) and not output_mismatches,
        "output_mismatch_prompt_indices": output_mismatches,
        "sync_actions": len(sync_actions),
        "async_actions": len(async_actions),
        "materializations_match": bool(sync_actions)
        and len(sync_actions) == len(async_actions)
        and not materialization_mismatches,
        "materialization_mismatch_action_indices": materialization_mismatches,
        "copy_ops_match": bool(sync_actions)
        and len(sync_actions) == len(async_actions)
        and not copy_op_mismatches,
        "copy_op_mismatch_action_indices": copy_op_mismatches,
        "transport_match": transport_match,
        "pipeline_counters": pipeline_counters,
        "pipeline_counters_passed": pipeline_counters_passed,
        "sync_prefill_tokens_per_s": sync_throughput,
        "async_prefill_tokens_per_s": async_throughput,
        "throughput_speedup_pct": (
            (async_throughput / sync_throughput - 1.0) * 100.0
            if async_throughput is not None and sync_throughput
            else None
        ),
        "sync_ttft_p50_s": sync_p50,
        "async_ttft_p50_s": async_p50,
        "ttft_p50_improvement_pct": (
            (1.0 - async_p50 / sync_p50) * 100.0
            if async_p50 is not None and sync_p50
            else None
        ),
        "sync_peak_memory_mib": int(
            (sync.get("memory") or {}).get("peak_memory_used_mb", 0)
        ),
        "async_peak_memory_mib": int(
            (async_row.get("memory") or {}).get("peak_memory_used_mb", 0)
        ),
        "async_overlap_ms": float(async_metrics.get("overlap_ms", 0.0)),
        "async_uncovered_boundary_tail_ms": float(
            async_metrics.get("uncovered_boundary_tail_ms", 0.0)
        ),
        "async_enqueue_tail_ms": float(
            async_metrics.get("pipeline_enqueue_tail_ms", 0.0)
        ),
    }


def pipeline_resource_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for row in rows:
        memory = row.get("memory") or {}
        baseline = int((memory.get("baseline") or {}).get("device_memory_used_mb", 0))
        process_peak = int(memory.get("peak_memory_used_mb", 0))
        device_peak = int(memory.get("peak_device_memory_used_mb", 0))
        external_peak = max(0, device_peak - process_peak)
        if baseline > 64 or external_peak > 64:
            violations.append(
                {
                    "run_name": row.get("run_name"),
                    "baseline_device_memory_mib": baseline,
                    "peak_device_minus_process_mib": external_peak,
                }
            )
    return {"passed": not violations, "violations": violations}


def build_pipeline_acceptance(
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    expected_pairs: int,
) -> dict[str, Any]:
    speedups = [
        float(pair["throughput_speedup_pct"])
        for pair in pairs
        if pair.get("throughput_speedup_pct") is not None
    ]
    ttft_improvements = [
        float(pair["ttft_p50_improvement_pct"])
        for pair in pairs
        if pair.get("ttft_p50_improvement_pct") is not None
    ]
    correctness_passed = len(pairs) == expected_pairs and all(
        pair["status_ok"]
        and pair["outputs_match"]
        and pair["materializations_match"]
        and pair["copy_ops_match"]
        and pair["transport_match"]
        and pair["pipeline_counters_passed"]
        for pair in pairs
    )
    resource_audit = pipeline_resource_audit(rows)
    formal_reasons = []
    formal_requirements = {
        "exclusive_resource_mode": args.resource_mode == "exclusive",
        "at_least_three_repeats": args.repeats >= 3,
        "sequence_length_at_least_2048": args.seq_len >= 2048,
        "at_least_eight_prompts": args.num_prompts >= 8,
        "concurrency_at_least_eight": args.concurrency >= 8,
        "warmup_at_least_eight_prompts": args.warmup_num_prompts >= 8,
        "pinned_host_tensors": bool(args.pin_cpu_tensors),
        "gpu_resource_audit": resource_audit["passed"],
    }
    for name, passed in formal_requirements.items():
        if not passed:
            formal_reasons.append(name)
    formal_eligible = all(formal_requirements.values())
    speedup_mean = mean(speedups) if speedups else None
    speedup_worst = min(speedups) if speedups else None
    performance_passed = (
        formal_eligible
        and len(speedups) == expected_pairs
        and speedup_mean is not None
        and speedup_mean > 0.0
        and speedup_worst is not None
        and speedup_worst >= 0.0
    )
    return {
        "complete": len(pairs) == expected_pairs,
        "expected_pairs": expected_pairs,
        "correctness_passed": correctness_passed,
        "formal_eligible": formal_eligible,
        "formal_requirement_results": formal_requirements,
        "formal_requirement_failures": formal_reasons,
        "resource_audit": resource_audit,
        "performance_passed": performance_passed,
        "passed": correctness_passed and performance_passed,
        "throughput_speedup_mean_pct": speedup_mean,
        "throughput_speedup_worst_pct": speedup_worst,
        "throughput_speedup_stddev_pct": (
            pstdev(speedups) if len(speedups) > 1 else 0.0 if speedups else None
        ),
        "ttft_p50_improvement_mean_pct": (
            mean(ttft_improvements) if ttft_improvements else None
        ),
        "ttft_p50_improvement_worst_pct": (
            min(ttft_improvements) if ttft_improvements else None
        ),
        "pairs": pairs,
    }


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    passed: bool,
) -> None:
    lines = [
        "# M20-B full/delta paired action-replay report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Run | Policy | Status | Prefill tok/s | TTFT proxy p50 (s) | Peak MiB | Actions |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        client = row.get("client") or {}
        memory = row.get("memory") or {}
        lines.append(
            f"| {row['run_name']} | "
            f"{row.get('policy_label', row.get('stage_policy', 'fifo'))} | "
            f"{row['status']} | "
            f"{client.get('prefill_tokens_per_s', '')} | "
            f"{client.get('latency_p50_s', '')} | "
            f"{memory.get('peak_memory_used_mb', '')} | {action_count(row)} |"
        )
    lines += ["", "## Replay checks", ""]
    trace_by_pair = {
        (int(trace["repeat"]), str(trace["policy"])): trace
        for trace in traces
    }
    for pair in pairs:
        trace = trace_by_pair[(int(pair["repeat"]), str(pair["policy"]))]
        lines.append(
            f"- repeat {pair['repeat']}: outputs_match={pair['outputs_match']}, "
            f"policy={pair['policy']}, "
            f"load_mode={pair['replay_load_mode']}, "
            f"actions={pair['replay_actions']}/{pair['expected_actions']}, "
            f"shared_actions={trace['shared_actions']}, "
            f"materialized={trace['materialized_actions']}/"
            f"{trace['planned_actions']}, "
            f"trace={trace['chain_hash']}"
        )
    lines += [
        "",
        f"Strict paired replay passed: `{passed}`.",
        "",
        "The full planner freezes action order and logical layer plans. The delta "
        "replay must execute those plans exactly while re-deriving physical "
        "retain/D2D/H2D operations.",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_pipeline_acceptance_report(
    output_dir: Path, result: dict[str, Any]
) -> None:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    lines = [
        "# M20-B Sync/Async Pipeline Performance Acceptance",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Correctness passed: `{result['correctness_passed']}`.  ",
        f"Formal eligible: `{result['formal_eligible']}`.  ",
        f"Performance passed: `{result['performance_passed']}`.  ",
        f"Overall passed: `{result['passed']}`.",
        "",
        "## Paired Repeats",
        "",
        "| Repeat | Policy | Sync tok/s | Async tok/s | Async gain | "
        "TTFT improvement | Actions | Copy ops | Pipeline counters |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for pair in result["pairs"]:
        lines.append(
            f"| {pair['repeat']} | {pair['policy']} | "
            f"{fmt(pair['sync_prefill_tokens_per_s'])} | "
            f"{fmt(pair['async_prefill_tokens_per_s'])} | "
            f"{fmt(pair['throughput_speedup_pct'])}% | "
            f"{fmt(pair['ttft_p50_improvement_pct'])}% | "
            f"{pair['sync_actions']}/{pair['async_actions']} | "
            f"{pair['copy_ops_match']} | "
            f"{pair['pipeline_counters_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Gate",
            "",
            f"- mean async throughput change: "
            f"`{fmt(result['throughput_speedup_mean_pct'])}%`",
            f"- weakest async throughput change: "
            f"`{fmt(result['throughput_speedup_worst_pct'])}%`",
            f"- speedup standard deviation: "
            f"`{fmt(result['throughput_speedup_stddev_pct'])}` percentage points",
            f"- mean TTFT p50 improvement: "
            f"`{fmt(result['ttft_p50_improvement_mean_pct'])}%`",
            f"- weakest TTFT p50 improvement: "
            f"`{fmt(result['ttft_p50_improvement_worst_pct'])}%`",
            "",
            "Acceptance requires exact sync/async outputs and materializations, zero "
            "pipeline errors/misses/blocks, exclusive formal resources, positive mean "
            "throughput change, and a non-negative weakest repeat.",
            "",
            "## Formal Requirements",
            "",
        ]
    )
    for name, passed in result["formal_requirement_results"].items():
        lines.append(f"- {name}: `{passed}`")
    if result["resource_audit"]["violations"]:
        lines.extend(["", "## Resource Violations", ""])
        lines.extend(
            f"- `{item['run_name']}`: baseline="
            f"{item['baseline_device_memory_mib']} MiB, device-process peak="
            f"{item['peak_device_minus_process_mib']} MiB"
            for item in result["resource_audit"]["violations"]
        )
    (output_dir / "PIPELINE_ACCEPTANCE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M20-B1a full planner and buffer-aware delta replay."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=common.MODEL)
    parser.add_argument("--gguf", type=Path, default=common.GGUF)
    parser.add_argument("--prompt-file", type=Path, default=common.WORKLOAD)
    parser.add_argument(
        "--prompt-identity-manifest", type=Path, default=common.IDENTITY
    )
    parser.add_argument("--oracle-trace", type=Path, default=common.ORACLE)
    parser.add_argument("--activation-stats", type=Path, default=common.ACTIVATION_STATS)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--max-replacements", type=int)
    parser.add_argument(
        "--placement",
        choices=["oracle", "static"],
        default="oracle",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["fifo", "min_delta", "cost_oracle"],
        default=["fifo"],
        help="Stage policies to run as individually paired full/delta replays.",
    )
    parser.add_argument(
        "--include-frequency-baseline",
        action="store_true",
        help=(
            "Add a matched-memory static frequency K=0 full/delta pair. "
            "The baseline uses FIFO stage ordering with zero replacements."
        ),
    )
    parser.add_argument("--cohort-size", type=int, default=2)
    parser.add_argument("--candidate-window", type=int, default=2)
    parser.add_argument("--max-consecutive", type=int, default=2)
    parser.add_argument("--max-wait-ms", type=float, default=0.0)
    parser.add_argument("--max-inflight-chunks", type=int, default=4)
    parser.add_argument("--stage-h2d-expert-ms", type=float, default=5.4)
    parser.add_argument("--stage-d2d-expert-ms", type=float, default=0.08)
    parser.add_argument("--stage-route-entry-gain-ms", type=float, default=0.0)
    parser.add_argument(
        "--stage-copy-contention-ms-per-expert", type=float, default=0.0
    )
    parser.add_argument("--stage-eviction-route-weight", type=float, default=0.0)
    parser.add_argument("--stage-queue-penalty-ms-per-s", type=float, default=0.0)
    parser.add_argument("--stage-min-gain-ms", type=float, default=0.0)
    parser.add_argument("--stage-confidence-threshold", type=float, default=1.0)
    parser.add_argument(
        "--require-physical-paths",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require both H2D and D2D in every delta run (B1a path proof).",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--replay-load-mode",
        choices=["sync", "async", "both"],
        default="sync",
        help=(
            "Materialization mode for the frozen replay. Use async to enable "
            "the ticket-aware A/B lookahead pipeline; sync is the matched B1b "
            "reference; both runs interleaved sync/async replays of each trace."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resource-mode",
        choices=["diagnostic", "exclusive"],
        default="diagnostic",
        help=(
            "Mark formal runs as exclusive and hold a host-wide runner lock. "
            "Exclusive mode also rejects SMT-overlapping CPU affinities."
        ),
    )
    parser.add_argument(
        "--exclusive-lock-path", type=Path, default=DEFAULT_EXCLUSIVE_LOCK
    )
    parser.add_argument(
        "--rotate-policy-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rotate adjacent full/delta policy pairs across repeats.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summary/correctness from completed run directories without launching servers.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--prompt-offset", type=int, default=128)
    parser.add_argument("--warmup-num-prompts", type=int, default=0)
    parser.add_argument("--warmup-prompt-offset", type=int, default=136)
    parser.add_argument("--warmup-concurrency", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--chunked-prefill-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--threadpool-count", type=int, default=2)
    parser.add_argument("--cpu-tensor-cache-items", type=int, default=256)
    parser.add_argument(
        "--pin-cpu-tensors", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--group-integrity-check", action="store_true")
    parser.add_argument("--mem-fraction-static", type=float, default=0.70)
    parser.add_argument("--max-total-tokens", type=int, default=40000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=31620)
    parser.add_argument("--server-timeout-s", type=float, default=600)
    parser.add_argument("--client-timeout-s", type=float, default=600)
    parser.add_argument(
        "--timing-log-interval",
        type=int,
        default=1,
        help="Emit one kt-time row every N per-layer calls; calibration requires 1.",
    )
    args = parser.parse_args()

    positive = {
        "group_size": args.group_size,
        "slots": args.slots,
        "cohort_size": args.cohort_size,
        "candidate_window": args.candidate_window,
        "max_consecutive": args.max_consecutive,
        "max_inflight_chunks": args.max_inflight_chunks,
        "repeats": args.repeats,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "chunked_prefill_size": args.chunked_prefill_size,
        "timing_log_interval": args.timing_log_interval,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        parser.error(f"parameters must be positive: {invalid}")
    if args.candidate_window < args.cohort_size:
        parser.error("--candidate-window must be at least --cohort-size")
    if args.max_wait_ms < 0:
        parser.error("--max-wait-ms cannot be negative")
    cost_values = {
        "stage_h2d_expert_ms": args.stage_h2d_expert_ms,
        "stage_d2d_expert_ms": args.stage_d2d_expert_ms,
        "stage_route_entry_gain_ms": args.stage_route_entry_gain_ms,
        "stage_copy_contention_ms_per_expert": (
            args.stage_copy_contention_ms_per_expert
        ),
        "stage_eviction_route_weight": args.stage_eviction_route_weight,
        "stage_queue_penalty_ms_per_s": args.stage_queue_penalty_ms_per_s,
    }
    invalid_costs = {key: value for key, value in cost_values.items() if value < 0}
    if invalid_costs:
        parser.error(f"cost parameters must be non-negative: {invalid_costs}")
    if not 0.0 <= args.stage_confidence_threshold <= 1.0:
        parser.error("--stage-confidence-threshold must be in [0, 1]")
    if args.max_replacements is None:
        args.max_replacements = args.slots
    if not 0 <= args.max_replacements <= args.slots:
        parser.error("--max-replacements must be in [0, --slots]")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stage_cohort_size = args.cohort_size
    args.stage_candidate_window = args.candidate_window
    args.stage_max_consecutive = args.max_consecutive
    args.stage_max_wait_ms = args.max_wait_ms
    args.stage_max_inflight_chunks = args.max_inflight_chunks
    args.action_replay_path = None
    if args.require_physical_paths is None:
        args.require_physical_paths = (
            args.policies == ["fifo"] and not args.include_frequency_baseline
        )
    return args


def build_specs(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    suffix = args.placement
    legacy_b1a = args.policies == ["fifo"] and not args.include_frequency_baseline
    replay_modes = (
        ["sync", "async"]
        if getattr(args, "replay_load_mode", "sync") == "both"
        else [str(getattr(args, "replay_load_mode", "sync"))]
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for policy in args.policies:
        full_mode = (
            f"b1_stage_full_{suffix}"
            if legacy_b1a
            else f"b1b_stage_full_{suffix}_{policy}"
        )
        delta_mode = (
            f"b1_stage_delta_replay_{suffix}"
            if legacy_b1a
            else f"b1b_stage_delta_replay_{suffix}_{policy}"
        )
        common_fields = {
            "group_size": args.group_size,
            "slots_per_layer": args.slots,
            "physical_slots": 2 * args.group_size * args.slots,
            "max_replacements": args.max_replacements,
            "stage_policy": policy,
            "policy_label": policy,
            "placement": args.placement,
        }
        result[policy] = [
            {
                **common_fields,
                "mode": full_mode,
                "materialization": "full",
                "action_role": "trace",
            },
        ]
        for replay_mode in replay_modes:
            result[policy].append(
                {
                    **common_fields,
                    "mode": delta_mode,
                    "materialization": "delta",
                    "action_role": "replay",
                    "load_mode": replay_mode,
                }
            )
    if args.include_frequency_baseline:
        # Keep the runtime policy as FIFO, but label the pair separately so
        # summary/calibration code cannot collide with oracle FIFO rows.
        result["frequency"] = [
            {
                "group_size": args.group_size,
                "slots_per_layer": args.slots,
                "physical_slots": 2 * args.group_size * args.slots,
                "max_replacements": 0,
                "stage_policy": "fifo",
                "policy_label": "frequency",
                "placement": "static",
                "mode": "b1b_stage_full_frequency",
                "materialization": "full",
                "action_role": "trace",
            },
        ]
        for replay_mode in replay_modes:
            result["frequency"].append(
                {
                    "group_size": args.group_size,
                    "slots_per_layer": args.slots,
                    "physical_slots": 2 * args.group_size * args.slots,
                    "max_replacements": 0,
                    "stage_policy": "fifo",
                    "policy_label": "frequency",
                    "placement": "static",
                    "mode": "b1b_stage_delta_replay_frequency",
                    "materialization": "delta",
                    "action_role": "replay",
                    "load_mode": replay_mode,
                }
            )
    return result


def spec_run_name(spec: dict[str, Any], repeat: int) -> str:
    run_name = (
        f"r{repeat}_{spec['mode']}_g{spec['group_size']}_s{spec['slots_per_layer']}"
    )
    if spec.get("max_replacements") is not None:
        run_name += f"_k{int(spec['max_replacements'])}"
    if str(spec.get("load_mode", "sync")) == "async":
        run_name += "_async"
    return run_name


def load_existing_row(
    output_dir: Path, spec: dict[str, Any], repeat: int
) -> dict[str, Any]:
    row = load_resumable_row(output_dir, spec, repeat)
    if row is None:
        run_name = spec_run_name(spec, repeat)
        status_path = output_dir / run_name / "runner_status.json"
        if not status_path.exists():
            raise FileNotFoundError(
                f"completed runner status is missing: {status_path}"
            )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"cannot summarize unsuccessful run {run_name}: "
            f"{status.get('error', status.get('status'))}"
        )
    return row


def load_resumable_row(
    output_dir: Path, spec: dict[str, Any], repeat: int
) -> dict[str, Any] | None:
    """Return a completed run row, or ``None`` when it must be executed.

    Resume is deliberately conservative: only a runner status explicitly marked
    ``ok`` is reusable. Failed or partially written rows fall through to the
    normal runner, which can recreate the result directory and its trace.
    """
    run_name = spec_run_name(spec, repeat)
    status_path = output_dir / run_name / "runner_status.json"
    if not status_path.exists():
        return None
    row = json.loads(status_path.read_text(encoding="utf-8"))
    if row.get("status") != "ok":
        return None
    return row


def main() -> int:
    args = parse_args()
    common.validate_inputs(args)
    specs_by_policy = build_specs(args)
    specs = [spec for pair_specs in specs_by_policy.values() for spec in pair_specs]
    policy_names = list(specs_by_policy)
    resource_snapshot = cpu_affinity_snapshot()
    provenance_path = args.output_dir / "provenance.json"
    generated_provenance = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner": str(RUNNER),
        "resource_snapshot": resource_snapshot,
        "policy_order_by_repeat": {
            str(repeat): policy_order_for_repeat(
                policy_names, repeat, rotate=args.rotate_policy_order
            )
            for repeat in range(1, args.repeats + 1)
        },
        "replay_order_by_repeat": {
            str(repeat): replay_modes_for_repeat(
                args.replay_load_mode, repeat
            )
            for repeat in range(1, args.repeats + 1)
        },
        "runtime_hashes": {
            str(path.relative_to(common.M20_ROOT)): common.sha256(path)
            for path in (
                common.M20_ROOT / "sglang/srt/layers/moe/kt_ep_wrapper.py",
                common.M20_ROOT / "sglang/srt/layers/moe/kt_group_expert_buffer.py",
                common.M20_ROOT / "sglang/srt/model_executor/model_runner.py",
                common.M20_ROOT / "sglang/srt/model_executor/kt_stage_batch.py",
                common.M20_ROOT / "sglang/srt/managers/scheduler.py",
                common.M20_ROOT / "sglang/srt/managers/kt_stage_scheduler.py",
                common.M20_ROOT / "sglang/srt/server_args.py",
                Path(common.__file__).resolve(),
                RUNNER,
            )
        },
        "asset_hashes": {
            "workload": common.sha256(args.prompt_file),
            "prompt_identity": common.sha256(args.prompt_identity_manifest),
            "oracle_trace": common.sha256(args.oracle_trace),
            "activation_stats": common.sha256(args.activation_stats),
            **common.model_metadata_hashes(args.model),
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "specs": specs,
    }
    if args.summarize_existing:
        if not provenance_path.exists():
            raise FileNotFoundError(
                f"existing provenance is missing: {provenance_path}"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    elif args.resume and provenance_path.exists():
        # Keep the original run contract when extending a partially completed
        # output directory. Rewriting provenance here would make a resumed
        # experiment appear to have a new timestamp/argument set.
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        provenance = generated_provenance
        common.write_json(provenance_path, provenance)
    if args.plan_only:
        print(json.dumps({"output_dir": str(args.output_dir), "specs": specs}, indent=2))
        return 0

    exclusive_lock = None
    if args.resource_mode == "exclusive" and not args.summarize_existing:
        exclusive_lock = acquire_exclusive_lock(
            args.exclusive_lock_path, resource_snapshot, args.cpu_threads
        )

    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    pipeline_pairs: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    port = args.base_port
    for repeat in range(1, args.repeats + 1):
        repeat_policy_order = policy_order_for_repeat(
            policy_names, repeat, rotate=args.rotate_policy_order
        )
        for policy in repeat_policy_order:
            pair_specs = specs_by_policy[policy]
            if args.summarize_existing:
                planner = load_existing_row(
                    args.output_dir, pair_specs[0], repeat
                )
            elif args.resume:
                planner = load_resumable_row(
                    args.output_dir, pair_specs[0], repeat
                )
                if planner is None:
                    while not common.port_available(port):
                        port += 1
                    planner = common.run_one(
                        args, pair_specs[0], repeat=repeat, port=port
                    )
                    port += 1
            else:
                while not common.port_available(port):
                    port += 1
                planner = common.run_one(
                    args, pair_specs[0], repeat=repeat, port=port
                )
                port += 1
            rows.append(planner)
            if planner.get("status") != "ok":
                continue

            trace_path = Path(planner["result_dir"]) / "actions.jsonl"
            trace_stats = load_trace_stats(trace_path, args.cohort_size)
            trace_stats["policy"] = policy
            trace_stats["repeat"] = repeat
            traces.append(trace_stats)
            args.action_replay_path = trace_path
            replay_spec_by_mode = {
                str(spec.get("load_mode", "sync")): spec
                for spec in pair_specs[1:]
            }
            replay_rows: dict[str, dict[str, Any]] = {}
            for replay_mode in replay_modes_for_repeat(
                args.replay_load_mode, repeat
            ):
                replay_spec = replay_spec_by_mode[replay_mode]
                if args.summarize_existing:
                    replay = load_existing_row(
                        args.output_dir, replay_spec, repeat
                    )
                elif args.resume:
                    replay = load_resumable_row(
                        args.output_dir, replay_spec, repeat
                    )
                    if replay is None:
                        while not common.port_available(port):
                            port += 1
                        replay = common.run_one(
                            args, replay_spec, repeat=repeat, port=port
                        )
                        port += 1
                else:
                    while not common.port_available(port):
                        port += 1
                    replay = common.run_one(
                        args, replay_spec, repeat=repeat, port=port
                    )
                    port += 1
                rows.append(replay)
                replay_rows[replay_mode] = replay
                pairs.append(compare_pair(planner, replay, trace_stats))
            if args.replay_load_mode == "both" and {
                "sync",
                "async",
            }.issubset(replay_rows):
                pipeline_pairs.append(
                    compare_pipeline_replays(
                        replay_rows["sync"], replay_rows["async"]
                    )
                )

    expected_pairs = args.repeats * sum(
        len(pair_specs) - 1 for pair_specs in specs_by_policy.values()
    )
    complete = len(pairs) == expected_pairs
    delta_rows = [
        row for row in rows if row.get("materialization") == "delta"
    ]
    delta_profiles = [row.get("group_profile") or {} for row in delta_rows]
    physical_paths_present = all(
        int(
            ((profile.get("last_action_metrics") or {}).get("d2d_experts", 0))
        ) > 0
        and int(
            ((profile.get("last_action_metrics") or {}).get("h2d_experts", 0))
        ) > 0
        for profile in delta_profiles
    ) if delta_profiles else False
    budget_audits = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        row_max_replacements = int(
            row.get("max_replacements", args.max_replacements)
        )
        budget_audits.append(
            {
                "run_name": row.get("run_name"),
                "max_replacements": row_max_replacements,
                **audit_replacement_budget(row, row_max_replacements),
            }
        )
    replacement_budget_passed = bool(budget_audits) and all(
        audit["passed"] for audit in budget_audits
    )
    # Different policies intentionally choose different placements. KT CPU
    # experts are Q8 while GPU experts are BF16, so cross-policy first tokens
    # are not a valid equality oracle. Correctness is the strict full/delta
    # replay comparison within each frozen action trace above.
    policy_outputs_match = all(pair["outputs_match"] for pair in pairs)
    physical_path_requirement_passed = (
        physical_paths_present if args.require_physical_paths else True
    )
    passed = (
        complete
        and all(row.get("status") == "ok" for row in rows)
        and all(pair["outputs_match"] for pair in pairs)
        and all(pair["action_counts_match"] for pair in pairs)
        and all(trace["materialization_complete"] for trace in traces)
        and all(trace["all_tickets_have_layer_plans"] for trace in traces)
        and policy_outputs_match
        and physical_path_requirement_passed
        and replacement_budget_passed
    )
    common.write_json(args.output_dir / "summary.json", rows)
    common.write_json(args.output_dir / "action_traces.json", traces)
    common.write_json(
        args.output_dir / "correctness.json",
        {
            "passed": passed,
            "complete": complete,
            "expected_pairs": expected_pairs,
            "physical_paths_present": physical_paths_present,
            "physical_paths_required": args.require_physical_paths,
            "physical_path_requirement_passed": physical_path_requirement_passed,
            "policy_outputs_match": policy_outputs_match,
            "cross_policy_output_comparison": "not_applicable_different_placement",
            "replacement_budget_passed": replacement_budget_passed,
            "replacement_budget_audits": budget_audits,
            "pairs": pairs,
        },
    )
    write_report(args.output_dir, rows, pairs, traces, passed)
    pipeline_acceptance = None
    if args.replay_load_mode == "both":
        expected_pipeline_pairs = args.repeats * len(specs_by_policy)
        pipeline_acceptance = build_pipeline_acceptance(
            pipeline_pairs,
            rows,
            args,
            expected_pipeline_pairs,
        )
        common.write_json(
            args.output_dir / "pipeline_acceptance.json",
            pipeline_acceptance,
        )
        write_pipeline_acceptance_report(
            args.output_dir, pipeline_acceptance
        )
    if exclusive_lock is not None:
        fcntl.flock(exclusive_lock.fileno(), fcntl.LOCK_UN)
        exclusive_lock.close()
    accepted = passed and (
        pipeline_acceptance is None or pipeline_acceptance["passed"]
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
