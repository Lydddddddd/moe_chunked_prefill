# M20-B Patch Manifest

`scripts/install_runtime.sh` installs and checks these source files against the
`sglang` package in the selected Python environment:

```text
sglang/srt/layers/moe/kt_ep_wrapper.py
sglang/srt/layers/moe/kt_group_expert_buffer.py
sglang/srt/model_executor/model_runner.py
sglang/srt/model_executor/kt_stage_batch.py
sglang/srt/managers/scheduler.py
sglang/srt/managers/kt_stage_scheduler.py
sglang/srt/server_args.py
```

The four files under `upstream_snapshot/` are the restore snapshot. The group
buffer and stage scheduler are M20-specific and remain in the package after a
restore.

## Freeze Rule

This file intentionally does not claim a runtime hash while source work is in
progress. Before a new formal experiment:

1. stop editing the seven patch files;
2. run `bash scripts/install_runtime.sh install` from `runtime/m20/` or the
   repository-relative equivalent;
3. run `bash scripts/install_runtime.sh check` and the full smoke suite;
4. record the resulting commit and per-file SHA-256 values in experiment
   provenance.

The included 2026-07-23 experiment keeps its own provenance and does not
certify source changes made afterward.
