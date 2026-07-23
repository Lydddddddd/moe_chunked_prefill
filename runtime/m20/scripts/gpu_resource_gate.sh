#!/usr/bin/env bash

wait_for_gpu_quiet() {
  if [[ "${WAIT_FOR_GPU_QUIET:-0}" != "1" ]]; then
    return
  fi

  local devices=${1:?GPU index or comma-separated GPU indices are required}
  local samples=${GPU_QUIET_SAMPLES:-10}
  local used_limit=${GPU_QUIET_USED_MIB:-64}
  local util_limit=${GPU_QUIET_UTIL_PCT:-5}
  local retry_seconds=${GPU_QUIET_RETRY_SECONDS:-30}
  local required_passes=${GPU_QUIET_REQUIRED_PASSES:-3}
  local quiet_passes=0
  local max_used max_util used util sample gpu
  local -a gpu_list
  IFS=, read -r -a gpu_list <<<"$devices"

  while true; do
    max_used=0
    max_util=0
    for ((i = 0; i < samples; i++)); do
      for gpu in "${gpu_list[@]}"; do
        if ! sample=$(nvidia-smi -i "$gpu" \
          --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null); then
          echo "gpu resource gate: nvidia-smi failed for GPU $gpu" >&2
          quiet_passes=0
          sleep "$retry_seconds"
          continue 3
        fi
        read -r used util < <(
          awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 + 0, $2 + 0}' \
            <<<"$sample"
        )
        ((used > max_used)) && max_used=$used
        ((util > max_util)) && max_util=$util
      done
      sleep 1
    done

    if ((max_used <= used_limit && max_util <= util_limit)); then
      quiet_passes=$((quiet_passes + 1))
    else
      quiet_passes=0
    fi
    printf 'gpu resource gate: gpus=%s max_used=%sMiB max_util=%s%% limits=%sMiB/%s%% quiet_passes=%s/%s\n' \
      "$devices" "$max_used" "$max_util" "$used_limit" "$util_limit" \
      "$quiet_passes" "$required_passes"
    if ((quiet_passes >= required_passes)); then
      return
    fi
    if ((quiet_passes == 0)); then
      sleep "$retry_seconds"
    fi
  done
}

start_gpu_isolation_monitor() {
  local target_gpu=${1:?target GPU index is required}
  local devices=${2:?comma-separated GPU indices are required}
  local log_path=${3:?monitor log path is required}
  local violation_path=${4:?violation log path is required}
  local used_limit=${GPU_RUNTIME_OTHER_USED_MIB:-64}
  local util_limit=${GPU_RUNTIME_OTHER_UTIL_PCT:-5}
  local interval=${GPU_RUNTIME_MONITOR_INTERVAL_SECONDS:-1}
  local -a gpu_list

  stop_gpu_isolation_monitor
  rm -f "$violation_path"
  printf 'timestamp_utc,gpu,memory_used_mib,utilization_pct\n' >"$log_path"
  IFS=, read -r -a gpu_list <<<"$devices"

  (
    trap 'exit 0' INT TERM
    while true; do
      local timestamp sample gpu used util
      timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
      for gpu in "${gpu_list[@]}"; do
        if ! sample=$(nvidia-smi -i "$gpu" \
          --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null); then
          printf '%s gpu=%s nvidia-smi_failed\n' "$timestamp" "$gpu" \
            >>"$violation_path"
          continue
        fi
        read -r used util < <(
          awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 + 0, $2 + 0}' \
            <<<"$sample"
        )
        printf '%s,%s,%s,%s\n' "$timestamp" "$gpu" "$used" "$util" \
          >>"$log_path"
        if [[ "$gpu" != "$target_gpu" ]] && \
          ((used > used_limit || util > util_limit)); then
          printf '%s gpu=%s used=%sMiB util=%s%% limits=%sMiB/%s%%\n' \
            "$timestamp" "$gpu" "$used" "$util" "$used_limit" "$util_limit" \
            >>"$violation_path"
        fi
      done
      sleep "$interval"
    done
  ) &
  GPU_ISOLATION_MONITOR_PID=$!
}

stop_gpu_isolation_monitor() {
  if [[ -n "${GPU_ISOLATION_MONITOR_PID:-}" ]]; then
    kill "$GPU_ISOLATION_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_ISOLATION_MONITOR_PID" 2>/dev/null || true
    GPU_ISOLATION_MONITOR_PID=
  fi
}
