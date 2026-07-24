# M20-B Group Double Buffer Runtime

`runtime/m20/` is the only active runtime track in this repository. It provides
stage-ready cross-request cohorts, immutable action tickets, strict full/delta
replay, and bounded A/B GPU expert-buffer materialization.

## Runtime Contract

```text
physical_slots = 2 x group_size x slots_per_layer
```

The installed patch supports TP=1, Qwen3 MoE, BF16 GPU experts, KT
LLAMAFILE/AVX2, and CUDA Graph disabled. The group buffer has two physical
buffers. An action computes on one buffer while the other may prepare a frozen
next ticket; the runtime commits only complete, version-matched materialization.

## Installation And Verification

The install script copies and compares exactly seven SGLang files. Use the
source tree as the authority; do not edit only site-packages.

```bash
bash runtime/m20/scripts/install_runtime.sh check
bash runtime/m20/scripts/run_m20_smoke_tests.sh
```

After a reviewed source change, install it explicitly:

```bash
bash runtime/m20/scripts/install_runtime.sh install
bash runtime/m20/scripts/install_runtime.sh check
```

`restore` replaces only the four files under `upstream_snapshot/`. It does not
delete M20-only files and is intended for debugging, not normal operation.

## Execution Modes

- Full planner: chooses cohorts and writes a hash-chained logical action trace.
- Delta replay: uses the frozen logical trace and re-derives physical
  retain/D2D/H2D operations.
- Async replay: prepares the next frozen ticket on the inactive buffer. It is
  limited to replay plus block-on-miss and must pass copy-operation audits.

The primary formal entry point is:

```bash
bash runtime/m20/scripts/run_m20_async_pipeline_formal.sh
```

It requires external model, GGUF, workload, identity manifest, oracle trace,
approved KT kernel patch, and exclusive GPU resources. It does not turn a
failed performance gate into an automatic retry.

## Acceptance Boundary

Smoke tests validate local contracts. A formal result additionally requires
matched source/install hashes, frozen provenance, full/replay output equality,
replacement-budget and physical-copy audits, exclusive resources, and three
paired repeats. See `docs/07_small_slot_stage_reuse_plan.md` for the complete
acceptance conditions.
