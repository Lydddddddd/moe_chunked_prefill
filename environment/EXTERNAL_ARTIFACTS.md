# External Reproduction Artifacts

The formal runner expects this external contract. Paths may be overridden with
environment variables, but the content must match.

## Large Assets

| Artifact | Expected size / identity |
|---|---|
| Model shards | 16 `model-*-of-00016.safetensors` files; 61,066,575,656 bytes total |
| Model config | SHA-256 `a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438` |
| Model index | SHA-256 `8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c` |
| Tokenizer | SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Tokenizer config | SHA-256 `a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3` |
| GGUF | 32,483,930,560 bytes; `Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf` |
| Workload | 5,631,378 bytes; SHA-256 `1b7ea6af331824b77c84a85840beac7bbb412ac12d7788958cf39b401d835b32` |
| Prompt identity | 88,546 bytes; SHA-256 `efe072e06c6f9b8cdddd835971f774f98d7330b00d88a11f4e82bb6ab0e2d234` |
| Oracle trace | 538,993,840 bytes; SHA-256 `6ec7591cac1b5274716bc50e6c65faa361655ae6dc7fed6d6c3be24da376dfb1` |
| Activation stats | 28,394 bytes; SHA-256 `493e5e4976a980f34497c56da4e2ba044d44a8d83b46c96027b12364fa81fd38` |

The GGUF and model shards should also travel with the artifact store's complete
SHA-256 manifest. They are not rehashed on every runner launch because reading
roughly 94 GB would distort setup time; the verifier checks their exact byte
contract and hashes the small identity files.

## kt-kernel Patch

The four Python modifications are checked into Git as
`environment/kt_kernel_patch/kt_kernel_0.5.0_m20.patch`. The validated
installation has these exact post-patch hashes:

| File | SHA-256 |
|---|---|
| `kt_kernel/__init__.py` | `2a2e1304a7dc054427008463af28440647d92f0b3a95c51626b86819c1841df2` |
| `kt_kernel/experts.py` | `c6e6390d1543b3d8c88fcfbe17314b001f96032349e7efaa25a7ef30fd463c88` |
| `kt_kernel/experts_base.py` | `ab022f6a862a63e179c45861f7e8ddc07b5818820c23aecf48453f566770ec59` |
| `kt_kernel/utils/llamafile.py` | `80b2b1ee502a4fa6f45029da68d8514e286e247ca8dd1d5721f0d195073dd5d8` |

The external runtime bundle contains:

| File | SHA-256 |
|---|---|
| `_kt_kernel_ext_avx2.cpython-310-x86_64-linux-gnu.so` | `066940ea0a97ab98bcfd369358a2be0178772381d3690b9870c679e963758d1f` |
| `libhwloc.so.15.5.2` | `2efe6e32d0d3454b2bce0d3b8cd437f48691475717be4b431f96bcc2d4d9bc54` |

This binary contract is specific to CPython 3.10, x86_64, AVX2, and the
validated Ubuntu 22.04 runtime. Rebuilding for a different ABI is a portability
test, not byte-identical reproduction.
