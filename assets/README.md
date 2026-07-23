# External Assets

`assets/` is intentionally not versioned except for this file. It contains
local symlinks or locally downloaded data required for end-to-end runs.

Required paths for the M20 formal runner are:

```text
assets/model_shim_qwen3/
assets/qwen3_gguf
assets/workloads/text/sharegpt_long_qwen3_min2048_512.jsonl
assets/workload_identity/sharegpt_long_qwen3_min2048_512.json
assets/kt_native_oracle_stats/
```

The model weights, GGUF, routed-expert traces, workload files, and third-party
sources must be obtained from the team's artifact store. Do not commit them to
GitHub. `environment/README.md` describes the runtime environment.

On the original workspace these paths are symlinks to the shared assets. A new
machine may either create equivalent links or set `MOE_ASSET_ROOT` to a
directory containing the same layout.
