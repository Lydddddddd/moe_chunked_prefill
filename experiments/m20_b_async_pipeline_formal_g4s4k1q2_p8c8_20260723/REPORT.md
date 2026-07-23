# M20-B full/delta paired action-replay report

Generated: 2026-07-23 13:50:41

| Run | Policy | Status | Prefill tok/s | TTFT proxy p50 (s) | Peak MiB | Actions |
|---|---|---|---:|---:|---:|---:|
| r1_b1b_stage_full_oracle_min_delta_g4_s4_k1 | min_delta | ok | 144.4271363138734 | 111.43103978410363 | 8150 | 509 |
| r1_b1b_stage_delta_replay_oracle_min_delta_g4_s4_k1 | min_delta | ok | 145.97609389940723 | 110.29912223946303 | 8150 | 509 |
| r1_b1b_stage_delta_replay_oracle_min_delta_g4_s4_k1_async | min_delta | ok | 145.99822098541043 | 110.21208440326154 | 8166 | 509 |

## Replay checks

- repeat 1: outputs_match=True, policy=min_delta, load_mode=sync, actions=509/509, shared_actions=489, materialized=509/509, trace=7377db67197abb3d7a2ac569e9284e2ed10bff1ff1e5ed56e4bf9b0a7c36bbcf
- repeat 1: outputs_match=True, policy=min_delta, load_mode=async, actions=509/509, shared_actions=489, materialized=509/509, trace=7377db67197abb3d7a2ac569e9284e2ed10bff1ff1e5ed56e4bf9b0a7c36bbcf

Strict paired replay passed: `True`.

The full planner freezes action order and logical layer plans. The delta replay must execute those plans exactly while re-deriving physical retain/D2D/H2D operations.
