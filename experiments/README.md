# Experiment Outputs

Generated experiment directories stay local or in the team's artifact store.
They contain server logs, request traces, CUDA memory samples, and summaries
that should not enter Git history.

The current formal target is M20-B `G4/S8/K1/Q2`, p8/c8, sequence length 2048,
with three paired full/sync/async repeats. Every result is authoritative only
for the commit, runtime hashes, assets, and hardware recorded in its
`provenance.json`.

For collaboration, return the compact report, provenance, correctness,
acceptance, and action-trace files listed in
`docs/11_EXTERNAL_REPRODUCTION_HANDOFF.md`. Store the complete output directory
as a checksummed archive outside GitHub.
