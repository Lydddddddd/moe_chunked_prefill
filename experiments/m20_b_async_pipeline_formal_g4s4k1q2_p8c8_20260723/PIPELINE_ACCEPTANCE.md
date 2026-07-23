# M20-B Sync/Async Pipeline Performance Acceptance

Generated: 2026-07-23 13:50:41

Correctness passed: `False`.  
Formal eligible: `False`.  
Performance passed: `False`.  
Overall passed: `False`.

## Paired Repeats

| Repeat | Policy | Sync tok/s | Async tok/s | Async gain | TTFT improvement | Actions | Copy ops | Pipeline counters |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | min_delta | 145.976 | 145.998 | 0.015% | 0.079% | 509/509 | True | True |

## Aggregate Gate

- mean async throughput change: `0.015%`
- weakest async throughput change: `0.015%`
- speedup standard deviation: `0.000` percentage points
- mean TTFT p50 improvement: `0.079%`
- weakest TTFT p50 improvement: `0.079%`

Acceptance requires exact sync/async outputs and materializations, zero pipeline errors/misses/blocks, exclusive formal resources, positive mean throughput change, and a non-negative weakest repeat.

## Formal Requirements

- exclusive_resource_mode: `True`
- at_least_three_repeats: `False`
- sequence_length_at_least_2048: `True`
- at_least_eight_prompts: `True`
- concurrency_at_least_eight: `True`
- warmup_at_least_eight_prompts: `True`
- pinned_host_tensors: `True`
- gpu_resource_audit: `True`
