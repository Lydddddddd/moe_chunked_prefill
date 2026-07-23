# Environment Contract

The validated environment used Python 3.10.14, `sglang-kt==0.6.3`,
`kt-kernel==0.5.0`, and `torch==2.9.1`. The package versions are recorded in
`requirements.txt`; the exact PyTorch wheel must match the target CUDA driver.

Create a base virtual environment with:

```bash
bash environment/bootstrap_env.sh
```

This does not by itself reproduce the validated runtime. Two local additions are
required before an end-to-end M20 run:

1. Install the team-approved `kt-kernel` patch. The validated environment has
   source changes in `kt_kernel/__init__.py`, `experts.py`, `experts_base.py`,
   and `utils/llamafile.py`, plus an AVX2 extension. This patch is not yet
   packaged as a clean installer, so obtain it from the team artifact store.
2. Provide the model, GGUF, workload, prompt-identity manifest, and oracle
   traces listed in `assets/README.md`.

After those are available, apply and check the versioned SGLang M20 patch:

```bash
bash runtime/m20/scripts/install_runtime.sh install
bash runtime/m20/scripts/install_runtime.sh check
for test in runtime/m20/tests/*.py; do
  .venv_kt/bin/python "$test"
done
```

Some hosts also require the bundled hwloc library before launching SGLang:

```bash
export LD_LIBRARY_PATH="/path/to/hwloc/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

The source patch is complete and checked into this repository. The kt-kernel
patch and large runtime assets remain external until they are frozen into a
reproducible artifact package.
