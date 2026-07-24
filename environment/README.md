# Environment Contract

The validated environment used Python 3.10.14, `sglang-kt==0.6.3`,
`kt-kernel==0.5.0`, and `torch==2.9.1`. The package versions are recorded in
`requirements.txt`; the exact PyTorch wheel must match the target CUDA driver.

Create a base virtual environment with:

```bash
bash environment/bootstrap_env.sh
```

This does not by itself reproduce the validated runtime. Two external inputs are
required before an end-to-end M20 run:

1. Extract the team-approved external runtime bundle under
   `environment/external_runtime/`, then run
   `bash environment/install_kt_kernel_patch.sh`. The four Python changes are
   versioned in Git; the bundle supplies the CPython 3.10 AVX2 extension and
   `libhwloc.so.15`.
2. Provide the model, GGUF, workload, prompt-identity manifest, and oracle
   traces listed in `assets/README.md`.

After those are available, apply and check the versioned SGLang M20 patch:

```bash
bash runtime/m20/scripts/install_runtime.sh install
bash runtime/m20/scripts/install_runtime.sh check
bash runtime/m20/scripts/run_m20_smoke_tests.sh
```

The project scripts automatically use the bundled hwloc library when it is
under `environment/external_runtime/lib`. For a custom extraction path, export:

```bash
export EXTERNAL_RUNTIME_DIR=/path/to/external_runtime
export LD_LIBRARY_PATH="$EXTERNAL_RUNTIME_DIR/lib:${LD_LIBRARY_PATH:-}"
```

The complete source patch is checked into this repository. Model weights,
GGUF, oracle data, and the platform-specific AVX2 extension remain external.

For cross-machine handoff, verify all dependency and artifact hashes before
running tests:

```bash
bash environment/verify_external_environment.sh
```

See `EXTERNAL_ARTIFACTS.md` and `docs/11_EXTERNAL_REPRODUCTION_HANDOFF.md` for
the exact delivery and return-artifact contract.
