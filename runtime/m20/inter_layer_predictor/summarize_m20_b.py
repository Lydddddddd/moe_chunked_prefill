#!/usr/bin/env python3
"""Aggregate repeated M20-B policy runs and enforce formal-result gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_POLICIES = {"frequency", "fifo", "min_delta", "cost_oracle"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def policy_label(row: Mapping[str, Any]) -> str:
    return str(row.get("policy_label", row.get("stage_policy", "fifo")))


def nested_number(row: Mapping[str, Any], *keys: str) -> float | None:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def final_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return (row.get("group_profile") or {}).get("last_action_metrics") or {}


def action_count(row: Mapping[str, Any]) -> int:
    return int(
        ((row.get("group_profile") or {}).get("event_counts") or {}).get(
            "action_end", 0
        )
    )


def mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(policy_label(row), str(row["materialization"]))].append(row)

    result: list[dict[str, Any]] = []
    for (policy, materialization), items in sorted(grouped.items()):
        throughputs = [
            value
            for row in items
            if (value := nested_number(row, "client", "prefill_tokens_per_s"))
            is not None
        ]
        p50s = [
            value
            for row in items
            if (value := nested_number(row, "client", "latency_p50_s"))
            is not None
        ]
        p95s = [
            value
            for row in items
            if (value := nested_number(row, "client", "latency_p95_s"))
            is not None
        ]
        peaks = [
            value
            for row in items
            if (value := nested_number(row, "memory", "peak_memory_used_mb"))
            is not None
        ]
        metrics = [final_metrics(row) for row in items]

        def metric_values(name: str) -> list[float]:
            return [float(item[name]) for item in metrics if item.get(name) is not None]

        actions = [float(action_count(row)) for row in items]
        result.append(
            {
                "policy": policy,
                "materialization": materialization,
                "repeat_count": len(items),
                "repeats": sorted(int(row["repeat"]) for row in items),
                "throughput_mean": mean(throughputs),
                "throughput_stdev": stdev(throughputs),
                "throughput_worst": min(throughputs) if throughputs else None,
                "ttft_p50_mean_s": mean(p50s),
                "ttft_p50_worst_s": max(p50s) if p50s else None,
                "ttft_p95_mean_s": mean(p95s),
                "gpu_peak_mean_mb": mean(peaks),
                "gpu_peak_max_mb": max(peaks) if peaks else None,
                "actions_mean": mean(actions),
                "h2d_bytes_mean": mean(metric_values("h2d_bytes")),
                "d2d_bytes_mean": mean(metric_values("d2d_bytes")),
                "zero_loads_mean": mean(metric_values("zero_loads")),
                "host_prepare_ms_mean": mean(metric_values("host_prepare_ms")),
                "copy_event_ms_mean": mean(metric_values("h2d_ms")),
            }
        )
    return result


def paired_delta_comparisons(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    delta = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("materialization") == "delta"
    ]
    frequency = {
        int(row["repeat"]): row
        for row in delta
        if policy_label(row) == "frequency"
    }
    result: list[dict[str, Any]] = []
    for row in delta:
        policy = policy_label(row)
        if policy == "frequency":
            continue
        baseline = frequency.get(int(row["repeat"]))
        if baseline is None:
            continue
        throughput = nested_number(row, "client", "prefill_tokens_per_s")
        baseline_throughput = nested_number(
            baseline, "client", "prefill_tokens_per_s"
        )
        p50 = nested_number(row, "client", "latency_p50_s")
        baseline_p50 = nested_number(baseline, "client", "latency_p50_s")
        h2d = final_metrics(row).get("h2d_bytes")
        baseline_h2d = final_metrics(baseline).get("h2d_bytes")
        result.append(
            {
                "repeat": int(row["repeat"]),
                "policy": policy,
                "throughput_speedup_pct": (
                    (throughput / baseline_throughput - 1.0) * 100.0
                    if throughput is not None and baseline_throughput
                    else None
                ),
                "ttft_p50_improvement_pct": (
                    (1.0 - p50 / baseline_p50) * 100.0
                    if p50 is not None and baseline_p50
                    else None
                ),
                "h2d_reduction_pct": (
                    (1.0 - float(h2d) / float(baseline_h2d)) * 100.0
                    if h2d is not None and baseline_h2d
                    else None
                ),
            }
        )
    return result


def aggregate_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["policy"])].append(row)
    result = []
    for policy, items in sorted(grouped.items()):
        speedups = [
            float(row["throughput_speedup_pct"])
            for row in items
            if row.get("throughput_speedup_pct") is not None
        ]
        ttft = [
            float(row["ttft_p50_improvement_pct"])
            for row in items
            if row.get("ttft_p50_improvement_pct") is not None
        ]
        h2d = [
            float(row["h2d_reduction_pct"])
            for row in items
            if row.get("h2d_reduction_pct") is not None
        ]
        result.append(
            {
                "policy": policy,
                "repeat_count": len(items),
                "throughput_speedup_mean_pct": mean(speedups),
                "throughput_speedup_worst_pct": min(speedups) if speedups else None,
                "ttft_p50_improvement_mean_pct": mean(ttft),
                "ttft_p50_improvement_worst_pct": min(ttft) if ttft else None,
                "h2d_reduction_mean_pct": mean(h2d),
            }
        )
    return result


def formal_gate(
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    correctness: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    args = (provenance.get("args") or {})
    reasons: list[str] = []
    if not correctness.get("passed"):
        reasons.append("strict full/delta correctness did not pass")
    if any(row.get("status") != "ok" for row in rows):
        reasons.append("one or more runs failed")
    if int(args.get("seq_len", 0)) < 2048:
        reasons.append("sequence length is below 2048")
    if int(args.get("num_prompts", 0)) < 8:
        reasons.append("fewer than 8 measured prompts")
    if int(args.get("repeats", 0)) < 3:
        reasons.append("fewer than 3 independent repeats")
    if args.get("resource_mode") != "exclusive":
        reasons.append("run was not recorded in exclusive resource mode")
    resource_snapshot = provenance.get("resource_snapshot") or {}
    if not resource_snapshot:
        reasons.append("CPU affinity snapshot is missing")
    elif resource_snapshot.get("has_smt_siblings"):
        reasons.append("CPU affinity includes SMT sibling overlap")
    if not args.get("rotate_policy_order"):
        reasons.append("policy pair order was not rotated across repeats")
    warmup_count = int(args.get("warmup_num_prompts", 0))
    if warmup_count <= 0:
        reasons.append("formal run has no warmup prompts")
    measured_prompts = set(
        range(
            int(args.get("prompt_offset", 0)),
            int(args.get("prompt_offset", 0)) + int(args.get("num_prompts", 0)),
        )
    )
    warmup_prompts = set(
        range(
            int(args.get("warmup_prompt_offset", 0)),
            int(args.get("warmup_prompt_offset", 0)) + warmup_count,
        )
    )
    if measured_prompts & warmup_prompts:
        reasons.append("measured and warmup prompt ranges overlap")
    observed = {policy_label(row) for row in rows}
    missing = sorted(REQUIRED_POLICIES - observed)
    if missing:
        reasons.append("missing policies: " + ", ".join(missing))
    for policy in sorted(REQUIRED_POLICIES):
        for materialization in ("full", "delta"):
            count = sum(
                policy_label(row) == policy
                and row.get("materialization") == materialization
                and row.get("status") == "ok"
                for row in rows
            )
            if count < 3:
                reasons.append(
                    f"{policy}/{materialization} has only {count} successful repeats"
                )
    occupied = [
        row.get("run_name")
        for row in rows
        if int(
            ((row.get("memory") or {}).get("baseline") or {}).get(
                "compute_process_count", 0
            )
        )
        > 0
    ]
    if occupied:
        reasons.append("external GPU compute processes were present at run start")
    calibration_authorized = bool(
        calibration and (calibration.get("gate") or {}).get("authorized")
    )
    return {
        "formal_run_eligible": not reasons,
        "calibration_authorized": calibration_authorized,
        "cost_oracle_conclusion_authorized": not reasons
        and calibration_authorized,
        "reasons": reasons,
        "calibration_reasons": (
            list((calibration.get("gate") or {}).get("reasons") or [])
            if calibration
            else ["calibration result was not provided"]
        ),
    }


def predictor_gate(
    formal: Mapping[str, Any], paired: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    reasons: list[str] = []
    if not formal.get("cost_oracle_conclusion_authorized"):
        reasons.append("formal run or calibrated cost-model gate is not authorized")
    cost = next((row for row in paired if row["policy"] == "cost_oracle"), None)
    if cost is None:
        reasons.append("cost-oracle has no matched frequency comparison")
    else:
        mean_speedup = cost.get("throughput_speedup_mean_pct")
        worst_speedup = cost.get("throughput_speedup_worst_pct")
        if mean_speedup is None or float(mean_speedup) < 3.0:
            reasons.append("cost-oracle mean throughput gain is below 3%")
        if worst_speedup is None or float(worst_speedup) < 0.0:
            reasons.append("cost-oracle worst-repeat throughput gain is negative")
    return {"proceed_to_predictor": not reasons, "reasons": reasons}


def fmt(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# M20-B repeated policy summary",
        "",
        f"Generated: {result['generated_at']}",
        "",
        f"Formal run eligible: `{result['formal_gate']['formal_run_eligible']}`.",
        f"Calibrated cost conclusion authorized: "
        f"`{result['formal_gate']['cost_oracle_conclusion_authorized']}`.",
        f"Proceed to predictor: `{result['predictor_gate']['proceed_to_predictor']}`.",
        "",
        "## Delta results",
        "",
        "| Policy | Repeats | tok/s mean | tok/s stddev | tok/s worst | TTFT p50 mean | H2D mean GB | D2D mean GB | Zero-load mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregates"]:
        if row["materialization"] != "delta":
            continue
        lines.append(
            f"| {row['policy']} | {row['repeat_count']} | "
            f"{fmt(row['throughput_mean'])} | {fmt(row['throughput_stdev'])} | "
            f"{fmt(row['throughput_worst'])} | "
            f"{fmt(row['ttft_p50_mean_s'])} | "
            f"{fmt(row['h2d_bytes_mean'] / 1e9 if row['h2d_bytes_mean'] is not None else None)} | "
            f"{fmt(row['d2d_bytes_mean'] / 1e9 if row['d2d_bytes_mean'] is not None else None)} | "
            f"{fmt(row['zero_loads_mean'])} |"
        )
    lines += [
        "",
        "## Versus frequency K=0",
        "",
        "| Policy | Throughput mean | Worst repeat | TTFT p50 mean | H2D reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["paired_aggregates"]:
        lines.append(
            f"| {row['policy']} | {fmt(row['throughput_speedup_mean_pct'])}% | "
            f"{fmt(row['throughput_speedup_worst_pct'])}% | "
            f"{fmt(row['ttft_p50_improvement_mean_pct'])}% | "
            f"{fmt(row['h2d_reduction_mean_pct'])}% |"
        )
    lines += ["", "## Gates", ""]
    gate_reasons = list(result["formal_gate"]["reasons"])
    calibration_reasons = list(result["formal_gate"]["calibration_reasons"])
    predictor_reasons = list(result["predictor_gate"]["reasons"])
    if not gate_reasons and not calibration_reasons and not predictor_reasons:
        lines.append("- all gates passed")
    else:
        lines.extend(f"- formal: {reason}" for reason in gate_reasons)
        lines.extend(f"- calibration: {reason}" for reason in calibration_reasons)
        lines.extend(f"- predictor: {reason}" for reason in predictor_reasons)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    args = parser.parse_args()
    args.experiment_dir = args.experiment_dir.resolve()
    if args.calibration_dir is not None:
        args.calibration_dir = args.calibration_dir.resolve()
    return args


def main() -> int:
    args = parse_args()
    rows = read_json(args.experiment_dir / "summary.json")
    provenance = read_json(args.experiment_dir / "provenance.json")
    correctness = read_json(args.experiment_dir / "correctness.json")
    calibration = (
        read_json(args.calibration_dir / "calibration.json")
        if args.calibration_dir is not None
        else None
    )
    aggregates = aggregate(rows)
    pairs = paired_delta_comparisons(rows)
    paired = aggregate_pairs(pairs)
    gate = formal_gate(rows, provenance, correctness, calibration)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summarizer": str(Path(__file__).resolve()),
        "summarizer_sha256": sha256(Path(__file__).resolve()),
        "experiment_dir": str(args.experiment_dir),
        "calibration_dir": (
            None if args.calibration_dir is None else str(args.calibration_dir)
        ),
        "aggregates": aggregates,
        "paired_delta_comparisons": pairs,
        "paired_aggregates": paired,
        "formal_gate": gate,
        "predictor_gate": predictor_gate(gate, paired),
    }
    write_json(args.experiment_dir / "m20_b_summary.json", result)
    write_report(args.experiment_dir / "M20_B_FORMAL_REPORT.md", result)
    print(
        json.dumps(
            {
                "output": str(args.experiment_dir / "m20_b_summary.json"),
                "formal_gate": result["formal_gate"],
                "predictor_gate": result["predictor_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
