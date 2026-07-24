# kt-kernel 0.5.0 M20 Patch

`kt_kernel_0.5.0_m20.patch` is the reviewable source delta applied after
installing the official CPython 3.10 x86_64 `kt-kernel==0.5.0` wheel.

`environment/install_kt_kernel_patch.sh` refuses to patch unknown source
hashes, applies this diff, installs the separately delivered AVX2 extension,
and verifies all five resulting files. The platform-specific extension and
`libhwloc.so.15` are intentionally carried in the external runtime artifact,
not in Git.
