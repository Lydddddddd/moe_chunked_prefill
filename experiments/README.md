# Experiment Artifacts

Raw experiment directories stay local or in the team's artifact store. They can
contain model outputs, server logs, request traces, CUDA memory samples, and
large JSON summaries that should not enter Git history.

The repository retains only the small acceptance records for the latest M20-B
run:

```text
m20_b_async_pipeline_formal_g4s4k1q2_p8c8_20260723/
  REPORT.md
  PIPELINE_ACCEPTANCE.md
  correctness.json
  pipeline_acceptance.json
  provenance.json
  action_traces.json
```

This 2026-07-23 run completed full, synchronous-delta, and asynchronous-delta
paths. The paired replay/output checks passed; it is a single-repeat functional
acceptance record, not a three-repeat performance claim.

The run's `provenance.json` remains authoritative. The local source changed
after this run, so the report must not be used as evidence for the newer source
until the runtime is reinstalled and revalidated.

When adding an experiment to GitHub, commit a compact report, provenance, and
correctness summary only. Put large traces, request JSONL, server logs, model
outputs, and summaries over 100 MB in the artifact store and record their
location and SHA-256 in the report.
