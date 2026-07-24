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

Exact sizes and known hashes for cross-machine reproduction are recorded in
`environment/EXTERNAL_ARTIFACTS.md`.

The required model shards and GGUF are roughly 94 GB together. Deliver this
directory through the team artifact store with its SHA-256 manifest; do not
bundle it with the Git repository or the small KT runtime package.

When copying from the original workspace, dereference symlinks (`rsync -aL` or
equivalent). The `qwen3_gguf` convenience path must resolve to a filename that
still ends in `.gguf`; otherwise set `GGUF` to the real file explicitly.
