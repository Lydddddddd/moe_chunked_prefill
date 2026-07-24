# Runtime Source

The active runtime source is `m20/`. It contains the complete M20-B Group
Double Buffer patch, installation script, tests, runners, and upstream restore
snapshot.

```bash
bash runtime/m20/scripts/install_runtime.sh check
bash runtime/m20/scripts/run_m20_smoke_tests.sh
```

Do not change the installed SGLang package without making the corresponding
change under `runtime/m20/`.
