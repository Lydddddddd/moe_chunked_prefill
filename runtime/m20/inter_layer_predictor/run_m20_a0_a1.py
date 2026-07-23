#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


M20_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = M20_ROOT.parents[1]
LEGACY_WORKSPACE = PROJECT_ROOT.parent
ASSET_ROOT = Path(os.environ.get("MOE_ASSET_ROOT", PROJECT_ROOT / "assets"))


def project_asset(relative: str, legacy_relative: str) -> Path:
    """Prefer the repository asset contract, retaining old-workspace support."""
    candidate = ASSET_ROOT / relative
    if candidate.exists() or candidate.is_symlink():
        return candidate
    return LEGACY_WORKSPACE / legacy_relative


def project_python() -> Path:
    for candidate in (
        PROJECT_ROOT / ".venv_kt" / "bin" / "python",
        LEGACY_WORKSPACE / ".venv_kt" / "bin" / "python",
    ):
        if candidate.is_file():
            return candidate
    return PROJECT_ROOT / ".venv_kt" / "bin" / "python"


PYTHON = project_python()
BENCHMARK = M20_ROOT / "inter_layer_predictor" / "benchmark_kt_prefill.py"
MODEL = project_asset("model_shim_qwen3", "moe_layered_prefill_system/assets/model_shim_qwen3")
GGUF = project_asset(
    "qwen3_gguf", "third_party/kt_weights/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf"
)
WORKLOAD = project_asset(
    "workloads/text/sharegpt_long_qwen3_min2048_512.jsonl",
    "workloads/text/sharegpt_long_qwen3_min2048_512.jsonl",
)
IDENTITY = project_asset(
    "workload_identity/sharegpt_long_qwen3_min2048_512.json",
    "outputs/kt_migration/workload_identity/sharegpt_long_qwen3_min2048_512.json",
)
ORACLE = project_asset(
    "kt_native_oracle_stats/kt_llamafile_tp1_seq2048_128p_gpu64_uniform_"
    "sharegpt_long_seq2048_test128_offset128_c4_cps256_top4rec/routed_experts_trace.jsonl",
    "outputs/kt_migration/kt_native_oracle_stats/"
    "kt_llamafile_tp1_seq2048_128p_gpu64_uniform_"
    "sharegpt_long_seq2048_test128_offset128_c4_cps256_top4rec/"
    "routed_experts_trace.jsonl",
)
ACTIVATION_STATS = project_asset(
    "kt_native_oracle_stats/sharegpt_long_seq2048_train128_per_layer_top4_activation_stats.pt",
    "outputs/kt_migration/kt_native_oracle_stats/"
    "sharegpt_long_seq2048_train128_per_layer_top4_activation_stats.pt",
)
HWLOC_LIB = project_asset(
    "hwloc_dev/root/usr/lib/x86_64-linux-gnu",
    "third_party/hwloc_dev/root/usr/lib/x86_64-linux-gnu",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata_hashes(model: Path) -> dict[str, str]:
    root = model if model.is_dir() else model.parent
    result: dict[str, str] = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = root / name
        if path.is_file():
            result[f"model/{name}"] = sha256(path)
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_server(port: int, timeout_s: float, proc: subprocess.Popen[Any]) -> bool:
    deadline = time.monotonic() + timeout_s
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
                if 200 <= response.status < 500:
                    return True
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ):
            pass
        time.sleep(2)
    return False


def stop_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def parse_compute_process_memory(output: str) -> tuple[int, int]:
    total_mb = 0
    count = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            int(fields[0])
            used_mb = int(float(fields[1]))
        except (TypeError, ValueError):
            continue
        total_mb += used_mb
        count += 1
    return total_mb, count


def query_gpu(gpu_index: int) -> dict[str, int]:
    device_proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    process_proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if device_proc.returncode != 0 or process_proc.returncode != 0:
        return {}
    try:
        device_used, total, util = [
            int(float(x.strip())) for x in device_proc.stdout.split(",")[:3]
        ]
    except (TypeError, ValueError):
        return {}
    process_used, process_count = parse_compute_process_memory(process_proc.stdout)
    return {
        # Preserve the report key while excluding pinned host mappings from
        # the attributable compute-process metric.
        "memory_used_mb": process_used,
        "device_memory_used_mb": device_used,
        "memory_total_mb": total,
        "utilization_pct": util,
        "compute_process_count": process_count,
    }


def start_gpu_monitor(result_dir: Path, gpu_index: int):
    stop = threading.Event()
    state: dict[str, Any] = {
        "phase": "startup",
        "baseline": query_gpu(gpu_index),
        "phase_peaks_mb": {},
        "device_phase_peaks_mb": {},
        "peak_memory_used_mb": 0,
        "peak_device_memory_used_mb": 0,
        "peak_utilization_pct": 0,
        "samples": 0,
    }
    trace_path = result_dir / "gpu_memory_trace.csv"

    def loop() -> None:
        with trace_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "time_unix,phase,process_memory_used_mb,device_memory_used_mb,"
                "memory_total_mb,utilization_pct,compute_process_count\n"
            )
            while not stop.is_set():
                sample = query_gpu(gpu_index)
                if sample:
                    phase = str(state["phase"])
                    used = int(sample["memory_used_mb"])
                    device_used = int(sample["device_memory_used_mb"])
                    state["samples"] += 1
                    state["peak_memory_used_mb"] = max(state["peak_memory_used_mb"], used)
                    state["peak_device_memory_used_mb"] = max(
                        state["peak_device_memory_used_mb"], device_used
                    )
                    state["peak_utilization_pct"] = max(
                        state["peak_utilization_pct"], int(sample["utilization_pct"])
                    )
                    peaks = state["phase_peaks_mb"]
                    peaks[phase] = max(int(peaks.get(phase, 0)), used)
                    device_peaks = state["device_phase_peaks_mb"]
                    device_peaks[phase] = max(
                        int(device_peaks.get(phase, 0)), device_used
                    )
                    handle.write(
                        f"{time.time():.6f},{phase},{used},"
                        f"{device_used},{sample['memory_total_mb']},"
                        f"{sample['utilization_pct']},"
                        f"{sample['compute_process_count']}\n"
                    )
                    handle.flush()
                stop.wait(0.5)

    thread = threading.Thread(target=loop, name="m20-gpu-monitor", daemon=True)
    thread.start()
    return stop, thread, state


def parse_group_log(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    timing: list[dict[str, float]] = []
    timing_re = re.compile(
        r"\[kt-time\].*?total=(?P<total>[0-9.]+)ms.*?"
        r"cpu_wait=(?P<cpu_wait>[0-9.]+)ms"
    )
    if not path.exists():
        return {"events": [], "event_counts": {}, "timing": {}}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "[kt-group] "
        if marker in line:
            try:
                events.append(json.loads(line.split(marker, 1)[1]))
            except json.JSONDecodeError:
                pass
        match = timing_re.search(line)
        if match:
            timing.append({key: float(value) for key, value in match.groupdict().items()})
    counts: dict[str, int] = {}
    for event in events:
        key = str(event.get("event", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    step_metrics = [
        event.get("metrics") for event in events if event.get("event") == "step_end"
    ]
    action_metrics = [
        event.get("metrics") for event in events if event.get("event") == "action_end"
    ]
    return {
        "event_counts": counts,
        "step_metrics": step_metrics,
        "last_step_metrics": step_metrics[-1] if step_metrics else None,
        "action_metrics": action_metrics,
        "last_action_metrics": action_metrics[-1] if action_metrics else None,
        "timing": {
            "samples": len(timing),
            "total_ms_sum": sum(row["total"] for row in timing),
            "cpu_wait_ms_sum": sum(row["cpu_wait"] for row in timing),
        },
    }


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    allowed = {
        "b0_static",
        "b0_oracle_foreground",
        "b1_static_split",
        "b2_group_sync",
        "b2_group_sync_static",
        "o1_async_block",
        "o1_async_fallback",
        "b0_stage_fifo",
        "b0_stage_fifo_static",
        "b0_stage_replay_static",
        "b1_stage_full_static",
        "b1_stage_delta_replay_static",
        "b1_stage_full_oracle",
        "b1_stage_delta_replay_oracle",
    }
    unknown = set(modes) - allowed
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    specs: list[dict[str, Any]] = []
    seen = set()
    for slots in args.slots:
        for group_size in args.group_sizes:
            for mode in modes:
                key = (
                    (mode, slots)
                    if mode in {"b0_static", "b0_oracle_foreground"}
                    else (mode, group_size, slots)
                )
                if key in seen:
                    continue
                seen.add(key)
                specs.append(
                    {
                        "mode": mode,
                        "group_size": group_size,
                        "slots_per_layer": slots,
                        "physical_slots": (
                            48 * slots
                            if mode in {"b0_static", "b0_oracle_foreground", "b1_static_split"}
                            else 2 * group_size * slots
                        ),
                    }
                )
    return specs


def server_command(args: argparse.Namespace, spec: dict[str, Any], port: int) -> list[str]:
    group_size = int(spec["group_size"])
    slots = int(spec["slots_per_layer"])
    cmd = [
        str(PYTHON),
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(args.model),
        "--kt-weight-path",
        str(args.gguf),
        "--kt-cpuinfer",
        str(args.cpu_threads),
        "--kt-threadpool-count",
        str(args.threadpool_count),
        "--kt-num-gpu-experts",
        str(slots),
        "--kt-method",
        "LLAMAFILE",
        "--kt-gpu-prefill-token-threshold",
        "4096",
        "--kt-expert-placement-strategy",
        "frequency",
        "--init-expert-location",
        str(args.activation_stats),
        "--attention-backend",
        "triton",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-running-requests",
        str(args.concurrency),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--watchdog-timeout",
        "3000",
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        "1",
        "--enable-p2p-check",
        "--served-model-name",
        "Qwen3",
        "--skip-server-warmup",
        "--disable-cuda-graph",
    ]
    mode = spec["mode"]
    stage_modes = {
        "b0_stage_fifo",
        "b0_stage_fifo_static",
        "b0_stage_replay_static",
        "b1_stage_full_static",
        "b1_stage_delta_replay_static",
        "b1_stage_full_oracle",
        "b1_stage_delta_replay_oracle",
    }
    is_stage_mode = mode in stage_modes or mode.startswith("b1b_stage_")
    if mode == "b0_oracle_foreground":
        cmd += ["--kt-enable-dynamic-expert-update"]
    elif mode == "b1_static_split":
        cmd += ["--kt-split-prefill-group-size", str(group_size)]
    elif (
        is_stage_mode
        or mode.startswith("b2_")
        or mode.startswith("o1_")
    ):
        load_mode = (
            "sync"
            if is_stage_mode
            or mode.startswith("b2_group_sync")
            else "async"
        )
        load_mode = str(spec.get("load_mode", load_mode))
        miss_policy = "cpu_fallback" if mode.endswith("fallback") else "block"
        prefetch_policy = str(
            spec.get(
                "placement", "static" if mode.endswith("static") else "oracle"
            )
        )
        materialization = str(
            spec.get(
                "materialization",
                "delta" if "b1_stage_delta_replay" in mode else "full",
            )
        )
        cmd += [
            "--kt-group-expert-buffer",
            "--kt-group-size",
            str(group_size),
            "--kt-slots-per-layer",
            str(slots),
            "--kt-group-buffer-count",
            "2",
            "--kt-group-load-mode",
            load_mode,
            "--kt-group-miss-policy",
            miss_policy,
            "--kt-group-prefetch-policy",
            prefetch_policy,
            "--kt-group-materialization",
            materialization,
        ]
        if spec.get("max_replacements") is not None:
            cmd += [
                "--kt-group-max-replacements",
                str(spec.get("max_replacements", slots)),
            ]
        if prefetch_policy == "oracle":
            cmd += [
                "--kt-group-oracle-required",
                "--kt-group-oracle-trace",
                str(args.oracle_trace),
                "--kt-group-oracle-prompt-identity-manifest",
                str(args.prompt_identity_manifest),
            ]
        if is_stage_mode:
            cmd += [
                "--kt-stage-ready-scheduler",
                "--kt-stage-policy",
                str(spec.get("stage_policy", "fifo")),
                "--kt-stage-cohort-size",
                str(args.stage_cohort_size),
                "--kt-stage-candidate-window",
                str(args.stage_candidate_window),
                "--kt-stage-max-consecutive",
                str(args.stage_max_consecutive),
                "--kt-stage-max-wait-ms",
                str(args.stage_max_wait_ms),
                "--kt-stage-max-inflight-chunks",
                str(args.stage_max_inflight_chunks),
                "--kt-stage-h2d-expert-ms",
                str(getattr(args, "stage_h2d_expert_ms", 5.4)),
                "--kt-stage-d2d-expert-ms",
                str(getattr(args, "stage_d2d_expert_ms", 0.08)),
                "--kt-stage-route-entry-gain-ms",
                str(getattr(args, "stage_route_entry_gain_ms", 0.0)),
                "--kt-stage-copy-contention-ms-per-expert",
                str(
                    getattr(
                        args, "stage_copy_contention_ms_per_expert", 0.0
                    )
                ),
                "--kt-stage-eviction-route-weight",
                str(getattr(args, "stage_eviction_route_weight", 0.0)),
                "--kt-stage-queue-penalty-ms-per-s",
                str(getattr(args, "stage_queue_penalty_ms_per_s", 0.0)),
                "--kt-stage-min-gain-ms",
                str(getattr(args, "stage_min_gain_ms", 0.0)),
                "--kt-stage-confidence-threshold",
                str(getattr(args, "stage_confidence_threshold", 1.0)),
            ]
    return cmd


def client_command(
    args: argparse.Namespace, result_dir: Path, port: int, run_name: str
) -> list[str]:
    return [
        str(PYTHON),
        str(BENCHMARK),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        "Qwen3",
        "--model-path",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--output-dir",
        str(result_dir),
        "--num-prompts",
        str(args.num_prompts),
        "--prompt-offset",
        str(args.prompt_offset),
        "--max-model-len",
        str(args.seq_len),
        "--max-tokens",
        "1",
        "--concurrency",
        str(args.concurrency),
        "--timeout-s",
        str(args.client_timeout_s),
        "--run-name",
        run_name,
        "--workload-id",
        "sharegpt_long_qwen3_min2048_512",
        "--prompt-identity-manifest",
        str(args.prompt_identity_manifest),
        "--warmup-num-prompts",
        str(args.warmup_num_prompts),
        "--warmup-prompt-offset",
        str(args.warmup_prompt_offset),
        "--warmup-concurrency",
        str(args.warmup_concurrency),
    ]


def runtime_env(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        env.pop(key, None)
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{HWLOC_LIB}:{old_ld}" if old_ld else str(HWLOC_LIB)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "CPUINFER_CPU_INSTRUCT": "AVX2",
            "CPUINFER_ENABLE_AMX": "OFF",
            "KT_KERNEL_CPU_VARIANT": "avx2",
            "SGLANG_KT_CPU_ROUTE_MASK": "1",
            "SGLANG_KT_RUNTIME_CPU_TENSOR_CACHE_ITEMS": str(args.cpu_tensor_cache_items),
            "SGLANG_KT_RUNTIME_PIN_CPU_TENSORS": "1" if args.pin_cpu_tensors else "0",
            "SGLANG_KT_RUNTIME_REGISTER_CPU_TENSORS": "0",
            "SGLANG_KT_RUNTIME_ASYNC_PREFETCH": (
                "1"
                if str(spec["mode"]).startswith("o1_async")
                or str(spec.get("load_mode", "sync")) == "async"
                else "0"
            ),
            "SGLANG_KT_RUNTIME_PREFETCH_WORKERS": "4",
            "SGLANG_KT_RUNTIME_PREFETCH_MAX_ITEMS": str(
                max(64, args.cpu_tensor_cache_items)
            ),
            "SGLANG_KT_HYBRID_TIMING": "1",
            "SGLANG_KT_HYBRID_TIMING_ROUTE_STATS": "0",
            "SGLANG_KT_HYBRID_TIMING_ALL_LAYERS": "1",
            "SGLANG_KT_HYBRID_TIMING_LOG_INTERVAL": str(
                getattr(args, "timing_log_interval", 16)
            ),
            "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_NUM_EXPERTS": "128",
            "SGLANG_KT_GROUP_INTEGRITY_CHECK": (
                "1" if args.group_integrity_check else "0"
            ),
        }
    )
    if spec["mode"] == "b0_oracle_foreground":
        env.update(
            {
                "SGLANG_KT_RUNTIME_FOREGROUND_ORACLE": "1",
                "SGLANG_KT_RUNTIME_ORACLE_PREFETCH": "1",
                "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_TRACE": str(args.oracle_trace),
                "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_PROMPT_IDENTITY_MANIFEST": str(
                    args.prompt_identity_manifest
                ),
                "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_REQUIRE_PROMPT_IDENTITY": "1",
                "SGLANG_KT_RUNTIME_SKIP_BATCH_SELECTION_WITH_ORACLE": "1",
                "SGLANG_KT_RUNTIME_MAPPING_FULL_COPY": "1",
                "SGLANG_KT_RUNTIME_PREFETCH_STAGE_SIZE": "1",
                "SGLANG_KT_RUNTIME_PREFETCH_TOP_K": str(spec["slots_per_layer"]),
                "SGLANG_KT_RUNTIME_CPU_PREFETCH_ENABLE": "0",
            }
        )
    return env


def run_one(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    repeat: int,
    port: int,
) -> dict[str, Any]:
    run_name = (
        f"r{repeat}_{spec['mode']}_g{spec['group_size']}_s{spec['slots_per_layer']}"
    )
    if spec.get("max_replacements") is not None:
        run_name += f"_k{int(spec['max_replacements'])}"
    if str(spec.get("load_mode", "sync")) == "async":
        run_name += "_async"
    result_dir = args.output_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    status_path = result_dir / "runner_status.json"
    if args.resume and status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            return existing
    server_log_path = result_dir / "server.log"
    client_log_path = result_dir / "client.log"
    server_cmd = server_command(args, spec, port)
    trace_mode = spec.get("action_role") == "trace" or spec["mode"] in {
        "b0_stage_fifo",
        "b0_stage_fifo_static",
        "b1_stage_full_static",
        "b1_stage_full_oracle",
    }
    replay_mode = spec.get("action_role") == "replay" or spec["mode"] in {
        "b0_stage_replay_static",
        "b1_stage_delta_replay_static",
        "b1_stage_delta_replay_oracle",
    }
    if trace_mode:
        action_trace_path = result_dir / "actions.jsonl"
        if action_trace_path.exists():
            action_trace_path.unlink()
        server_cmd += [
            "--kt-action-trace-path",
            str(action_trace_path),
        ]
    elif replay_mode:
        if args.action_replay_path is None:
            raise ValueError(
                f"{spec['mode']} requires --action-replay-path"
            )
        server_cmd += [
            "--kt-action-replay-path",
            str(args.action_replay_path),
        ]
    client_cmd = client_command(args, result_dir, port, run_name)
    env = runtime_env(args, spec)
    recorded_env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "CPUINFER_CPU_INSTRUCT",
        "CPUINFER_ENABLE_AMX",
        "KT_KERNEL_CPU_VARIANT",
        "SGLANG_KT_CPU_ROUTE_MASK",
        "SGLANG_KT_RUNTIME_CPU_TENSOR_CACHE_ITEMS",
        "SGLANG_KT_RUNTIME_PIN_CPU_TENSORS",
        "SGLANG_KT_RUNTIME_REGISTER_CPU_TENSORS",
        "SGLANG_KT_RUNTIME_ASYNC_PREFETCH",
        "SGLANG_KT_RUNTIME_PREFETCH_WORKERS",
        "SGLANG_KT_RUNTIME_PREFETCH_MAX_ITEMS",
        "SGLANG_KT_HYBRID_TIMING",
        "SGLANG_KT_HYBRID_TIMING_LOG_INTERVAL",
        "SGLANG_KT_RUNTIME_ORACLE_PREFETCH_NUM_EXPERTS",
        "SGLANG_KT_GROUP_INTEGRITY_CHECK",
    )
    config = {
        **spec,
        "repeat": repeat,
        "port": port,
        "server_command": server_cmd,
        "client_command": client_cmd,
        "runtime_environment": {
            key: env[key] for key in recorded_env_keys if key in env
        },
    }
    write_json(result_dir / "config.json", config)
    stop, monitor, memory = start_gpu_monitor(result_dir, args.gpu)
    started = time.time()
    status = "failed"
    error = ""
    server_proc: subprocess.Popen[Any] | None = None
    try:
        with server_log_path.open("w", encoding="utf-8") as server_log:
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            if not wait_server(port, args.server_timeout_s, server_proc):
                raise RuntimeError(
                    f"server failed to become ready, returncode={server_proc.poll()}"
                )
            memory["phase"] = "ready_idle"
            time.sleep(2)
            memory["phase"] = "benchmark"
            with client_log_path.open("w", encoding="utf-8") as client_log:
                client = subprocess.run(
                    client_cmd,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.client_timeout_s + 60,
                    check=False,
                )
            if client.returncode != 0:
                raise RuntimeError(f"benchmark failed with returncode={client.returncode}")
            status = "ok"
    except Exception as exc:  # noqa: BLE001 - persist all experiment failures.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        memory["phase"] = "shutdown"
        if server_proc is not None:
            stop_process(server_proc)
        stop.set()
        monitor.join(timeout=5)
        baseline_used = int(memory.get("baseline", {}).get("memory_used_mb", 0))
        memory["peak_increment_mb"] = max(
            0, int(memory.get("peak_memory_used_mb", 0)) - baseline_used
        )
        write_json(result_dir / "gpu_memory_summary.json", memory)

    summary_path = result_dir / "default_summary.json"
    client_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    group_profile = parse_group_log(server_log_path)
    write_json(result_dir / "group_profile.json", group_profile)
    row = {
        **spec,
        "repeat": repeat,
        "run_name": run_name,
        "status": status,
        "error": error,
        "elapsed_s": time.time() - started,
        "client": client_summary,
        "memory": memory,
        "group_profile": group_profile,
        "result_dir": str(result_dir),
    }
    write_json(status_path, row)
    return row


def validate_correctness_matrix(rows: list[dict[str, Any]]) -> None:
    """Reject incomplete reference sets before reporting a run as correct."""

    active_modes = {str(row["mode"]) for row in rows}
    needs_oracle = bool(
        active_modes
        & {
            "b0_oracle_foreground",
            "b2_group_sync",
            "o1_async_block",
            "o1_async_fallback",
            "b0_stage_fifo",
        }
    )
    if needs_oracle and "b0_oracle_foreground" not in active_modes:
        raise ValueError(
            "oracle group modes require b0_oracle_foreground in --modes for "
            "same-placement output comparison"
        )
    static_candidates = active_modes & {
        "b1_static_split",
        "b0_stage_fifo_static",
    }
    if static_candidates and "b0_static" not in active_modes:
        raise ValueError(
            f"{sorted(static_candidates)} require b0_static in --modes for "
            "same-placement output comparison"
        )
    if (
        "b0_stage_replay_static" in active_modes
        and "b0_stage_fifo_static" not in active_modes
    ):
        raise ValueError(
            "b0_stage_replay_static requires b0_stage_fifo_static in --modes"
        )


def compare_outputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    skipped = [
        {
            "repeat": int(row["repeat"]),
            "candidate": row["run_name"],
            "reason": f"run status is {row.get('status')}: {row.get('error', '')}",
        }
        for row in rows
        if row.get("status") != "ok"
    ]
    for row in rows:
        if row.get("status") == "ok":
            by_repeat.setdefault(int(row["repeat"]), []).append(row)
    comparisons = []
    all_match = not skipped
    for repeat, repeat_rows in sorted(by_repeat.items()):
        for row in repeat_rows:
            needs_oracle_reference = row["mode"] in {
                "b0_oracle_foreground",
                "b2_group_sync",
                "o1_async_block",
                "o1_async_fallback",
                "b0_stage_fifo",
            }
            if row["mode"] == "b0_stage_replay_static":
                baseline_mode = "b0_stage_fifo_static"
            else:
                baseline_mode = (
                    "b0_oracle_foreground"
                    if needs_oracle_reference
                    else "b0_static"
                )
            slots = int(row["slots_per_layer"])
            baseline = next(
                (
                    candidate
                    for candidate in repeat_rows
                    if candidate["mode"] == baseline_mode
                    and int(candidate["slots_per_layer"]) == slots
                ),
                None,
            )
            if baseline is None:
                skipped.append(
                    {
                        "repeat": repeat,
                        "candidate": row["run_name"],
                        "reason": f"missing {baseline_mode} with slots_per_layer={slots}",
                    }
                )
                continue
            base_path = Path(baseline["result_dir"]) / "default_request_metrics.jsonl"
            base_values = {
                int(item["prompt_index"]): str(item.get("generated_text", ""))
                for item in map(
                    json.loads,
                    base_path.read_text(encoding="utf-8").splitlines(),
                )
            }
            path = Path(row["result_dir"]) / "default_request_metrics.jsonl"
            values = {
                int(item["prompt_index"]): str(item.get("generated_text", ""))
                for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            }
            mismatches = sorted(
                key for key in set(base_values) | set(values) if base_values.get(key) != values.get(key)
            )
            match = not mismatches
            all_match = all_match and match
            comparisons.append(
                {
                    "repeat": repeat,
                    "baseline": baseline["run_name"],
                    "candidate": row["run_name"],
                    "reference_kind": (
                        "same_oracle_placement"
                        if needs_oracle_reference
                        else "same_static_placement"
                    ),
                    "match": match,
                    "mismatch_prompt_indices": mismatches,
                }
            )
    return {
        "all_match": bool(comparisons) and all_match,
        "complete": not skipped,
        "evaluated_count": len(comparisons),
        "skipped": skipped,
        "comparisons": comparisons,
    }


def write_report(output_dir: Path, rows: list[dict[str, Any]], correctness: dict[str, Any]) -> None:
    lines = [
        "# M20-A0/A1 运行报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 配置 | 状态 | TTFT p50 (s) | prefill tok/s | GPU peak (MiB) | physical slots |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        client = row.get("client") or {}
        memory = row.get("memory") or {}
        lines.append(
            f"| {row['run_name']} | {row['status']} | "
            f"{client.get('latency_p50_s', '')} | {client.get('prefill_tokens_per_s', '')} | "
            f"{memory.get('peak_memory_used_mb', '')} | {row['physical_slots']} |"
        )
    lines += [
        "",
        "## 正确性",
        "",
        "同 placement 参考下首 token 全部一致："
        f"`{correctness.get('all_match')}`；覆盖完整："
        f"`{correctness.get('complete')}`。",
        "",
        "单轮结果只用于诊断；正式性能结论必须使用至少三轮交错配对。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.model,
        args.gguf,
        args.prompt_file,
        args.prompt_identity_manifest,
        args.oracle_trace,
        args.activation_stats,
        BENCHMARK,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if not port_available(args.base_port):
        raise RuntimeError(f"base port is already in use: {args.base_port}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M20-A0/A1 controlled experiments.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--gguf", type=Path, default=GGUF)
    parser.add_argument("--prompt-file", type=Path, default=WORKLOAD)
    parser.add_argument("--prompt-identity-manifest", type=Path, default=IDENTITY)
    parser.add_argument("--oracle-trace", type=Path, default=ORACLE)
    parser.add_argument("--action-replay-path", type=Path)
    parser.add_argument("--activation-stats", type=Path, default=ACTIVATION_STATS)
    parser.add_argument("--group-sizes", type=csv_ints, default=[2])
    parser.add_argument("--slots", type=csv_ints, default=[4])
    parser.add_argument(
        "--modes",
        default=(
            "b0_static,b0_oracle_foreground,b1_static_split,"
            "b2_group_sync,o1_async_block,o1_async_fallback"
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prompt-offset", type=int, default=128)
    parser.add_argument("--warmup-num-prompts", type=int, default=8)
    parser.add_argument("--warmup-prompt-offset", type=int, default=136)
    parser.add_argument("--warmup-concurrency", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--chunked-prefill-size", type=int, default=256)
    parser.add_argument("--stage-cohort-size", type=int, default=1)
    parser.add_argument("--stage-candidate-window", type=int, default=1)
    parser.add_argument("--stage-max-consecutive", type=int, default=2)
    parser.add_argument("--stage-max-wait-ms", type=float, default=0.0)
    parser.add_argument("--stage-max-inflight-chunks", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--threadpool-count", type=int, default=2)
    parser.add_argument("--cpu-tensor-cache-items", type=int, default=256)
    parser.add_argument(
        "--group-integrity-check",
        action="store_true",
        help="Verify sampled group payloads and mapping replicas after each group commit.",
    )
    parser.add_argument(
        "--pin-cpu-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use page-locked host expert tensors. Enabled by default because "
            "O1 async H2D is not valid with pageable sources."
        ),
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.70)
    parser.add_argument("--max-total-tokens", type=int, default=40000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=31420)
    parser.add_argument("--server-timeout-s", type=float, default=600)
    parser.add_argument("--client-timeout-s", type=float, default=600)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.stage_cohort_size <= 0:
        parser.error("--stage-cohort-size must be positive")
    if args.stage_candidate_window < args.stage_cohort_size:
        parser.error("--stage-candidate-window must be at least cohort size")
    if args.stage_max_consecutive <= 0 or args.stage_max_inflight_chunks <= 0:
        parser.error("stage fairness and inflight bounds must be positive")
    if args.stage_max_wait_ms < 0:
        parser.error("--stage-max-wait-ms cannot be negative")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_inputs(args)
    specs = build_specs(args)
    validate_correctness_matrix(specs)
    provenance = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_hashes": {
            str(path.relative_to(M20_ROOT)): sha256(path)
            for path in (
                M20_ROOT / "sglang/srt/layers/moe/kt_ep_wrapper.py",
                M20_ROOT / "sglang/srt/layers/moe/kt_group_expert_buffer.py",
                M20_ROOT / "sglang/srt/model_executor/model_runner.py",
                M20_ROOT / "sglang/srt/model_executor/kt_stage_batch.py",
                M20_ROOT / "sglang/srt/managers/scheduler.py",
                M20_ROOT / "sglang/srt/managers/kt_stage_scheduler.py",
                M20_ROOT / "sglang/srt/server_args.py",
            )
        },
        "asset_hashes": {
            "workload": sha256(args.prompt_file),
            "prompt_identity": sha256(args.prompt_identity_manifest),
            "oracle_trace": sha256(args.oracle_trace),
            "activation_stats": sha256(args.activation_stats),
            **model_metadata_hashes(args.model),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "specs": specs,
    }
    write_json(args.output_dir / "provenance.json", provenance)
    if args.plan_only:
        print(json.dumps({"output_dir": str(args.output_dir), "specs": specs}, indent=2))
        return 0

    rows = []
    port = args.base_port
    for repeat in range(1, args.repeats + 1):
        ordered = specs if repeat % 2 else list(reversed(specs))
        for spec in ordered:
            while not port_available(port):
                port += 1
            print(
                f"[m20] repeat={repeat} mode={spec['mode']} "
                f"g={spec['group_size']} s={spec['slots_per_layer']} port={port}",
                flush=True,
            )
            rows.append(run_one(args, spec, repeat=repeat, port=port))
            port += 1
    write_json(args.output_dir / "summary.json", rows)
    correctness = compare_outputs(rows)
    write_json(args.output_dir / "correctness.json", correctness)
    write_report(args.output_dir, rows, correctness)
    return (
        0
        if all(row["status"] == "ok" for row in rows)
        and correctness["all_match"]
        and correctness["complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
