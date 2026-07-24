# Current M20 Repository

This repository is intentionally limited to the active M20-B Group Double
Buffer implementation. It does not contain historical M16-M19 per-layer
prefetch, Resident-Delta, M19 baseline, or legacy predictor code.

```text
.
├── README.md
├── docs/
│   ├── 00_REPOSITORY_LAYOUT.md
│   ├── 07_small_slot_stage_reuse_plan.md
│   ├── 07_IMPLEMENTATION_FILE_MAP.md
│   └── 09_end_to_end_inference_flow.md
├── runtime/m20/
│   ├── sglang/                 Seven installed SGLang patch files
│   ├── inter_layer_predictor/  M20 execution and analysis tools
│   ├── scripts/                Installation and formal-run entry points
│   ├── tests/                  Smoke and CUDA-pipeline tests
│   └── upstream_snapshot/      Restore snapshot for four upstream files
├── environment/                Dependency, KT-patch, and artifact contract
├── assets/README.md            Non-Git model/trace/workload contract
└── experiments/                Local outputs; only the policy README is tracked
```

## Team Workflow

1. Read `07_small_slot_stage_reuse_plan.md` before changing the runtime.
2. Use `07_IMPLEMENTATION_FILE_MAP.md` to find ownership and tests.
3. Make changes in `runtime/m20/`, not only in site-packages.
4. Run `install_runtime.sh check` and `run_m20_smoke_tests.sh` before a GPU
   experiment.
5. Commit only compact reports, provenance, and correctness records. Keep
   model weights, GGUF, traces, request logs, and large summaries in the
   artifact store.

The active installed patch consists of the seven files in
`runtime/m20/PATCH_MANIFEST.md`.
