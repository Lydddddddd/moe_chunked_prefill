# M20-B Implementation Map

This file maps the active implementation of
[`07_small_slot_stage_reuse_plan.md`](07_small_slot_stage_reuse_plan.md). The
current path is M20-B Group Double Buffer with stage-ready cohorts and bounded
delta materialization.

## Execution Path

```text
server_args.py
  -> scheduler.py
     -> kt_stage_scheduler.py
     -> kt_stage_batch.py
        -> model_runner.py
           -> kt_group_expert_buffer.py
              -> kt_ep_wrapper.py
```

The scheduler forms a cohort from ready chunks in one stage, freezes an
immutable action ticket, and uses that ticket to select and materialize a
physical A/B group buffer. The wrapper executes GPU hits and KT CPU fallback;
the batch layer demultiplexes results back to individual requests.

## Installed Runtime Patch

`runtime/m20/scripts/install_runtime.sh` installs and checks these seven files:

| File | Responsibility |
|---|---|
| `sglang/srt/managers/kt_stage_scheduler.py` | ready queues, tickets, policy, replay, K budget |
| `sglang/srt/managers/scheduler.py` | SGLang event-loop integration and cohort lifecycle |
| `sglang/srt/model_executor/kt_stage_batch.py` | cohort pack/demux and metadata validation |
| `sglang/srt/model_executor/model_runner.py` | ticket metadata and group action lifecycle |
| `sglang/srt/layers/moe/kt_group_expert_buffer.py` | `[2][G][S]` storage, retain, D2D, H2D, delta and async handoff |
| `sglang/srt/layers/moe/kt_ep_wrapper.py` | oracle demand lookup and hybrid GPU/CPU MoE execution |
| `sglang/srt/server_args.py` | feature flags, capacity and replay contract |

`upstream_snapshot/` contains four upstream files used by `restore`. It does
not remove the two M20-only files, so use restore only for debugging.

## M20 Tools

| Path | Use |
|---|---|
| `inter_layer_predictor/benchmark_kt_prefill.py` | request benchmark and metrics collection |
| `inter_layer_predictor/run_m20_a0_a1.py` | common server/client orchestration and provenance |
| `inter_layer_predictor/run_m20_b.py` | full planner plus strict delta replay |
| `inter_layer_predictor/calibrate_m20_b_cost.py` | route, transport and preparation cost calibration |
| `inter_layer_predictor/summarize_m20_b.py` | correctness and formal-run summary |
| `scripts/install_runtime.sh` | install, check, and restore runtime patch |
| `scripts/run_m20_async_pipeline_formal.sh` | frozen-trace sync/async formal acceptance |

## Test Map

| Test | Scope |
|---|---|
| `kt_group_buffer_smoke.py` | capacity, lifecycle, full/delta, retain, D2D/H2D, K budget |
| `kt_runtime_mapping_smoke.py` | logical expert to physical slot mapping |
| `m20_stage_scheduler_smoke.py` | queues, tickets, policy, replay and fairness |
| `m20_stage_batch_smoke.py` | cohort pack/demux and metadata |
| `m20_stage_runtime_smoke.py` | scheduler event-loop integration |
| `m20_server_args_smoke.py` | feature flags and parameter bounds |
| `m20_runner_logic_smoke.py` | common orchestration and reference logic |
| `m20_b_runner_smoke.py` | paired replay and copy-operation audit |
| `m20_b_summary_smoke.py` | formal result gate |
| `m20_cost_calibration_smoke.py` | calibration and holdout gate |
| `m20_pipeline_cuda_smoke.py` | CUDA copy stream, event and async ticket handoff |

Run them from the repository root:

```bash
bash runtime/m20/scripts/install_runtime.sh check
for test in runtime/m20/tests/*.py; do
  .venv_kt/bin/python "$test"
done
```

## External Requirements

End-to-end execution needs an external model shim, GGUF, workload,
prompt-identity manifest, routed-expert oracle trace, activation stats, and the
approved KT kernel patch. See `assets/README.md` and `environment/README.md`.
These assets are not committed to GitHub.

## Current Boundary

The implemented active path is `R=0` Group Double Buffer. Predictor shadow and
predictor action are not in the runtime. Async replay is implemented and has
smoke coverage; a formal performance claim requires frozen source, a matching
installed runtime, isolated three-repeat evidence, and the acceptance gates in
the 07 design document.
