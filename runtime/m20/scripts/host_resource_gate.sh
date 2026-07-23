#!/usr/bin/env bash

wait_for_host_quiet() {
  if [[ "${WAIT_FOR_HOST_QUIET:-0}" != "1" ]]; then
    return
  fi

  local samples=${HOST_QUIET_SAMPLES:-5}
  local busy_limit=${HOST_QUIET_BUSY_PCT:-10}
  local runnable_limit=${HOST_QUIET_MAX_RUNNABLE:-32}
  local retry_seconds=${HOST_QUIET_RETRY_SECONDS:-30}
  local required_passes=${HOST_QUIET_REQUIRED_PASSES:-1}
  local quiet_passes=0
  local max_busy max_r

  while true; do
    read -r max_busy max_r < <(
      vmstat 1 "$((samples + 1))" | awk '
        NR > 3 {
          busy = $13 + $14
          if (busy > max_busy) max_busy = busy
          if ($1 > max_r) max_r = $1
        }
        END { print max_busy + 0, max_r + 0 }
      '
    )
    if ((max_busy <= busy_limit && max_r <= runnable_limit)); then
      quiet_passes=$((quiet_passes + 1))
    else
      quiet_passes=0
    fi
    printf 'host resource gate: max_busy=%s%% max_r=%s limits=%s%%/%s quiet_passes=%s/%s\n' \
      "$max_busy" "$max_r" "$busy_limit" "$runnable_limit" \
      "$quiet_passes" "$required_passes"
    if ((quiet_passes >= required_passes)); then
      return
    fi
    sleep "$retry_seconds"
  done
}
