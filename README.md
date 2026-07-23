# M20 Group Double Buffer Prefill

This repository contains only the current implementation of M20-B: a
stage-ready MoE prefill scheduler with bounded A/B GPU expert buffers.

The active path is:

```text
ready chunks
  -> stage-ready cohort selection
  -> immutable action ticket
  -> full or delta A/B buffer materialization
  -> hybrid GPU-hit / CPU-fallback MoE execution
  -> demux and advance the request
```

The scheduler supports FIFO, `min_delta`, and `cost_oracle` choices. The active
research configuration uses `G=4` consecutive layers, a bounded number of
slots per layer, and `K=1` replacement per layer action. The async replay path
prepares an arbitrary next frozen ticket while the current ticket computes.

## Start Here

1. [Current design and acceptance gates](docs/07_small_slot_stage_reuse_plan.md)
2. [Implementation map and ownership](docs/07_IMPLEMENTATION_FILE_MAP.md)
3. [End-to-end execution flow](docs/09_end_to_end_inference_flow.md)
4. [Repository layout](docs/00_REPOSITORY_LAYOUT.md)
5. [Environment contract](environment/README.md)

## Repository Layout

```text
runtime/m20/                 Active implementation
  sglang/                    Seven SGLang runtime patch files
  inter_layer_predictor/     M20 runner, benchmark, cost calibration, reports
  scripts/                   Runtime install and formal async experiment entry
  tests/                     Unit, integration, and CUDA pipeline smoke tests
  upstream_snapshot/         Four files used by restore
docs/                         Current design, file map, and execution flow
environment/                 Package versions and bootstrap instructions
assets/README.md             Required non-Git model and trace assets
experiments/                 Only compact current acceptance records
```

Historical M16-M19 prefetch, Resident-Delta, M19 baseline, and prior predictor
work remain in the original local workspace but are deliberately excluded from
this collaboration repository.

## Setup And Verification

Provision the external assets and KT patch described by
[`environment/README.md`](environment/README.md), then run from the repository
root:

```bash
bash runtime/m20/scripts/install_runtime.sh check
for test in runtime/m20/tests/*.py; do
  .venv_kt/bin/python "$test"
done
```

`runtime/m20/scripts/run_m20_async_pipeline_formal.sh` is the formal M20-B
async replay entry point. It requires the external model, GGUF, workload,
identity manifest, oracle trace, and an exclusive GPU.

## Current Source Status

The local validation environment currently matches the versioned M20 runtime.
Always rerun `install_runtime.sh check` after changing any of the seven patch
files and before a new GPU experiment. The included 2026-07-23 acceptance
record is tied to its own `provenance.json`.
