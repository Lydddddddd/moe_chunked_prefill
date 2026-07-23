# Runtime Source

The active runtime source is `m20/`. It contains the complete M20-B Group
Double Buffer patch, installation script, tests, runners, and upstream restore
snapshot.

```bash
bash runtime/m20/scripts/install_runtime.sh check
for test in runtime/m20/tests/*.py; do
  .venv_kt/bin/python "$test"
done
```

Do not change the installed SGLang package without making the corresponding
change under `runtime/m20/`.
