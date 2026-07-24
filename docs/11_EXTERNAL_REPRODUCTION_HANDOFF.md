# M20-B 跨机器复测交付手册

本手册用于把当前 M20-B Group Double Buffer 实现交给另一台机器复测。只验证当前
`G4/S8/K1/Q2` 的 full、sync delta 和 async delta 三条路径，不重跑历史路线。

## 1. 交付三件套

### A. GitHub 代码

```text
https://github.com/Lydddddddd/moe_chunked_prefill
```

交付时必须给出固定 commit。仓库包含七个 SGLang runtime patch、四个 `kt-kernel`
源码 patch、实验 runner、11 项 smoke、环境校验脚本和本文档。不要让复测方直接跑浮动的
`main`。

### B. 小型外部 runtime 包

从原验证机生成：

```bash
bash environment/package_external_runtime.sh
```

将生成的以下两个文件放到团队共享存储：

```text
m20_external_runtime_cp310_x86_64.tar.zst
m20_external_runtime_cp310_x86_64.tar.zst.sha256
```

包内只有验证过的 CPython 3.10/x86_64 AVX2 extension 和 `libhwloc.so.15`，约数 MB。
四个可审查的 Python 源码改动已经在 Git，不在二进制包中重复保存。

### C. 大资产目录

以下约 95 GB 内容走团队共享存储，不进入 GitHub：

```text
model_shim_qwen3/                         16 个 safetensors shards + config/index/tokenizer
qwen3_gguf                               Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
workloads/text/sharegpt_long_qwen3_min2048_512.jsonl
workload_identity/sharegpt_long_qwen3_min2048_512.json
kt_native_oracle_stats/
  kt_llamafile_tp1_seq2048_128p_gpu64_uniform_sharegpt_long_seq2048_test128_offset128_c4_cps256_top4rec/
    routed_experts_trace.jsonl
  sharegpt_long_seq2048_train128_per_layer_top4_activation_stats.pt
```

精确大小和 hash 见 `environment/EXTERNAL_ARTIFACTS.md`。`.venv_kt` 不交付，复测方自行
创建；没有相同的 KT patch、workload 和 oracle，只能称为移植测试，不能称为复现。

转存资产时必须解引用原机软链接，例如使用 `rsync -aL`；不能把指向原机 `/data` 或
`../outputs` 的链接原样发走。`qwen3_gguf` 应是指向完整 `.gguf` 文件名的链接，或者在
运行时直接设置 `GGUF=/shared/.../Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf`。

## 2. 机器要求

原验证机器：NVIDIA RTX A6000 48 GiB、2 x AMD EPYC 7532、64 个物理核、Ubuntu
22.04、Python 3.10.14、NVIDIA driver 570.124.06。

不同 GPU/CPU 可以验证正确性和同机 paired speedup，但不能直接比较绝对 tok/s、TTFT
或显存。正式性能复现建议至少 48 GiB GPU、64 个独占物理 CPU cores，并用 `lscpu
-e=CPU,CORE,SOCKET,NODE` 选择每个物理 core 的一个硬件线程作为 `CPUSET`。

系统工具至少需要 `python3.10-venv`、`patch`、`jq`、`zstd`、`libnuma1`、NVIDIA
driver 和 CUDA-compatible PyTorch 环境。

## 3. 从零准备

```bash
git clone https://github.com/Lydddddddd/moe_chunked_prefill.git
cd moe_chunked_prefill
git checkout <交付 commit>

bash environment/bootstrap_env.sh

(cd /shared && \
  sha256sum -c m20_external_runtime_cp310_x86_64.tar.zst.sha256)
tar --zstd -xf /shared/m20_external_runtime_cp310_x86_64.tar.zst \
  -C environment
export LD_LIBRARY_PATH="$PWD/environment/external_runtime/lib:${LD_LIBRARY_PATH:-}"

bash environment/install_kt_kernel_patch.sh
```

把大资产目录链接到 `assets/` 的约定路径；也可以不建链接，而将包含相同布局的目录传给
`MOE_ASSET_ROOT`：

```bash
export MOE_ASSET_ROOT=/shared/m20_assets
bash environment/verify_external_environment.sh
```

任何 hash、模型 shard 数、Python ABI 或依赖版本不匹配，都应先停止，不要带着错误
环境跑正式实验。

## 4. 安装与 preflight

```bash
bash runtime/m20/scripts/install_runtime.sh install
bash runtime/m20/scripts/install_runtime.sh check

GPU=0 \
OUTPUT_DIR="$PWD/reproduction/preflight" \
  bash runtime/m20/scripts/run_external_preflight.sh
```

通过标准：安装的七个 SGLang 文件与 Git 完全一致，固定的 11 个 M20-B smoke 全部
PASS，CUDA pipeline smoke 不能 SKIP。日志保存在 `reproduction/preflight/`。

机器资源已经确认独占后，也可以用一个入口依次执行 preflight、1 轮筛选和 3 轮正式
复测：

```bash
GPU=0 CPUSET=0-63 CPU_THREADS=64 \
  bash runtime/m20/scripts/run_external_reproduction.sh
```

一键入口会把三部分结果放在同一个带 UTC 时间戳的 `reproduction/m20_external_*`
目录下；回传时以终端打印的实际 `RUN_ROOT` 为准。

## 5. 一轮正确性筛选

先用独立输出目录验证 full/sync/async 三条路径，预计约 20--30 分钟：

```bash
GPU=0 \
CPUSET=0-63 \
CPU_THREADS=64 \
REPEATS=1 \
OUTPUT_DIR="$PWD/reproduction/m20_g4s8k1q2_screen" \
  bash runtime/m20/scripts/run_m20_async_pipeline_formal.sh
```

```bash
jq -e '.complete and .correctness_passed and .resource_audit.passed and
       (.expected_pairs == 1)' \
  reproduction/m20_g4s8k1q2_screen/pipeline_acceptance.json
```

还必须确认请求输出、action count、frozen plan、materialization、transport 和 copy-op
审计一致。原机已有一轮参考：sync `142.954 tok/s`、async `155.744 tok/s`、async
`+8.947%`，正确性通过；这只有一轮，不是正式性能结论。

## 6. 三轮正式复测

筛选通过后使用全新目录，不复用筛选 provenance。预计约 60--90 分钟：

```bash
GPU=0 \
CPUSET=0-63 \
CPU_THREADS=64 \
REPEATS=3 \
OUTPUT_DIR="$PWD/reproduction/m20_g4s8k1q2_formal_r3" \
  bash runtime/m20/scripts/run_m20_async_pipeline_formal.sh
```

功能复现通过标准：

```bash
jq -e '.complete and .correctness_passed and .formal_eligible and
       .resource_audit.passed and (.expected_pairs == 3)' \
  reproduction/m20_g4s8k1q2_formal_r3/pipeline_acceptance.json
```

性能结论单独看 `.performance_passed`：三轮平均 async throughput 必须为正，且最弱一轮
不得为负。负结果也是有效实验结果，所以 runner 会正常结束；不能只看 shell 退出码。

## 7. 不需要重跑

- M16-M19 per-layer prefetch 和 Resident-Delta。
- S12/S16 大容量诊断。
- 历史 Q4/K sweep。
- M20-C corruption/predictor。

## 8. 回传内容

先回传这些小文件，便于快速验收：

```text
reproduction/preflight/system_info.txt
reproduction/preflight/packages.txt
reproduction/preflight/runtime_sha256.txt
reproduction/preflight/kt_kernel_sha256.txt
reproduction/preflight/preflight.log
reproduction/m20_g4s8k1q2_formal_r3/provenance.json
reproduction/m20_g4s8k1q2_formal_r3/REPORT.md
reproduction/m20_g4s8k1q2_formal_r3/PIPELINE_ACCEPTANCE.md
reproduction/m20_g4s8k1q2_formal_r3/correctness.json
reproduction/m20_g4s8k1q2_formal_r3/pipeline_acceptance.json
reproduction/m20_g4s8k1q2_formal_r3/action_traces.json
```

完整正式目录打包到 artifact store，不要提交 GitHub：

```bash
tar -I 'zstd -T0 -10' -cf m20_g4s8k1q2_formal_r3.tar.zst \
  reproduction/m20_g4s8k1q2_formal_r3
sha256sum m20_g4s8k1q2_formal_r3.tar.zst \
  > m20_g4s8k1q2_formal_r3.tar.zst.sha256
```

完整包应保留 server logs、request metrics、runner status、group profile、GPU memory
trace 和 frozen action trace。
