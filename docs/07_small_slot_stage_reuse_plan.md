# M20-B Stage-Ready Group Double Buffer

## Goal

M20-B improves long-context MoE prefill under a small GPU expert budget. It
does not reorder layers within one request. Instead, it chooses among chunks
from different requests whose dependencies have already reached the same layer
group, so that one physical expert working set can serve a cohort before it is
replaced.

The active path is:

```text
ready chunk states
  -> choose one stage and a bounded cohort
  -> freeze an immutable action ticket
  -> materialize the target group in A/B physical buffers
  -> execute the group with GPU hits and KT CPU fallback
  -> demux, advance state, and publish the next ready group
```

## Scope

- Runtime: `runtime/m20/` only.
- Model execution: TP=1, Qwen3 MoE, BF16 GPU experts, KT LLAMAFILE/AVX2,
  CUDA Graph disabled.
- Active implementation: `R=0` Group Double Buffer. There is no permanent
  resident source cache in this repository.
- The predictor is not connected to the runtime. Oracle corruption tooling is
  offline only and is not a scheduler action path.

## Physical Contract

For group size `G`, slots per layer `S`, and two physical buffers:

```text
physical_slots = 2 x G x S
```

Each buffer stores contiguous layer-local expert weights for its target group.
The active buffer cannot be modified while its group computes. The inactive
buffer may prepare the next frozen ticket. At a group boundary the runtime
commits only a complete, version-matched target buffer.

The action plan supports:

- retain: keep an expert already present in the target buffer;
- D2D: copy a compatible expert from the other buffer;
- H2D: prepare and load a CPU-resident expert;
- zero-load: represent an empty slot explicitly.

For each layer, the planner enforces `changed <= K`. This logical replacement
budget is independent from the physical source selected by materialization.

## Stage-Ready Scheduling

Each request chunk has a `ChunkStageState`. A state becomes ready only after its
previous group has completed, so the scheduler never violates a request's layer
order or KV-cache dependency.

The scheduler:

1. chooses a nonempty stage-ready queue;
2. considers a bounded candidate window;
3. forms a cohort up to `C` chunks from one stage;
4. scores FIFO, `min_delta`, or `cost_oracle` choices;
5. freezes a `NextActionTicket` with logical layer plans and copy operations;
6. dispatches the cohort, then demultiplexes it back to request states.

`max_consecutive` and `max_wait_ms` prevent one ready stage from starving the
others. A request has at most one in-flight chunk in the current path.

## Full, Delta, And Async Replay

The full planner is the reference path: it creates a hash-chained action trace.
Delta replay uses the exact same logical ticket order and layer plans while
re-deriving retain/D2D/H2D operations from physical buffer state. Output,
action count, replacement budget, and trace identity must match.

Async replay is restricted to frozen replay plus block-on-miss. It prepares an
arbitrary next ticket on the inactive buffer while the current ticket computes.
The pending record is adopted only when ticket identity, target buffer version,
and copy completion agree; otherwise the runtime drains and falls back safely.

## Current Candidate And Gates

The current target configuration is a `G=4` group with bounded slots, `K=1`,
and stage-ready `min_delta` selection. The formal async script uses a frozen
full trace and paired sync/async replay. Exact `S`, cohort, and candidate-window
settings are recorded in each experiment provenance.

Before a performance claim, all of the following are required:

1. The source is frozen and `install_runtime.sh check` matches the installed
   SGLang runtime.
2. The M20 smoke suite passes, including the CUDA pipeline smoke.
3. Full planner and sync/async replay produce identical request outputs and
   frozen action plans.
4. Copy-operation, replacement-budget, resource-exclusivity, and provenance
   audits pass.
5. Three isolated paired repeats meet the configured formal gate. A one-repeat
   functional acceptance result is not a performance conclusion.

## Team Entry Points

- [Implementation map](07_IMPLEMENTATION_FILE_MAP.md)
- [End-to-end flow](09_end_to_end_inference_flow.md)
- [Runtime operation](../runtime/m20/README.md)
- [External environment](../environment/README.md)
