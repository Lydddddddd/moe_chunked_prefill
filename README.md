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
6. [External reproduction handoff](docs/11_EXTERNAL_REPRODUCTION_HANDOFF.md)

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
  kt_kernel_patch/           Reviewable kt-kernel 0.5.0 source diff
assets/README.md             Required non-Git model and trace assets
experiments/                 Local outputs; only the policy README is tracked
```

Historical M16-M19 prefetch, Resident-Delta, M19 baseline, and prior predictor
work remain in the original local workspace but are deliberately excluded from
this collaboration repository.

## Setup And Verification

Provision the external assets and KT patch described by
[`environment/README.md`](environment/README.md), then run from the repository
root:

```bash
bash environment/verify_external_environment.sh
bash runtime/m20/scripts/install_runtime.sh install
bash runtime/m20/scripts/install_runtime.sh check
bash runtime/m20/scripts/run_m20_smoke_tests.sh
```

`runtime/m20/scripts/run_m20_async_pipeline_formal.sh` is the formal M20-B
async replay entry point. It requires the external model, GGUF, workload,
identity manifest, oracle trace, and an exclusive GPU.

For a clean handoff, follow `docs/11_EXTERNAL_REPRODUCTION_HANDOFF.md`. After
setup, `runtime/m20/scripts/run_external_reproduction.sh` runs the fixed
preflight, one-repeat screen, and three-repeat formal suite in order.

## Current Source Status

The local validation environment currently matches the versioned M20 runtime.
Always rerun `install_runtime.sh check` after changing any of the seven patch
files and before a new GPU experiment. Generated evidence stays local or in the
team artifact store and is tied to its own `provenance.json`.
