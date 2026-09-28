# FLUX.3 Action on RTX 5090

本目录同步 2026-09-28 完成验证的单卡 RTX 5090 实现：混合精度投影 +
SageAttention2，以及之后保持该基线输出的无损优化。
它作为独立实验入口提供，未替换仓库原有的 GB200 部署路径。

## 实测结果

| 固定输入 | 原始 BF16 median | 固定 Sage2 median | 最终 ABBA median | 相对原始 BF16 |
|---|---:|---:|---:|---:|
| 0 | 1396.406 ms | 722.625 ms | 684.603 ms | 2.040× |
| 1 | 1408.608 ms | 724.123 ms | 685.827 ms | 2.054× |

最终一轮相对固定 Sage2 基线的延迟下降为 **5.26–5.29%**。每路径/输入
5 次预热，15 组 ABBA（每路径 30 次计时），120 次请求的动作输出全部逐位一致。
原始 BF16 与最终结果来自不同进程；ABBA 只用于衡量最后一轮无损优化。

计时为 warm `predict_action_chunk`，包括图像/VAE conditioning 与全部四步
视频/动作去噪；不包括权重加载、Qwen 文本编码、首次编译、图捕获及预热。
独立进程最终入口测得 674.105 / 680.339 ms，但主结论采用上面的 ABBA 数据。

相对原始 BF16 的 joint MAE 为 0.009818 / 0.015379 rad，joint max 为
0.046994 / 0.045333 rad，gripper max 为 0.007246 / 0.010859。
两个输入均通过原阈值：joint MAE ≤ 0.02 rad、joint max ≤ 0.05 rad、
gripper max ≤ 0.05，以及相对 recorded action 的 joint MAE 退化 ≤ 0.02 rad。
这只是两组固定离线观测的数值筛查，不是机器人成功率评估。

**“无损”仅指最后一轮相对固定 Sage2 混合精度基线没有新增差异。**
整条路线包含 FP8 投影与 SageAttention2 近似，不与原始 BF16 逐位相同。

## 实现与来源

| 文件 | 作用 |
|---|---|
| `sm120_optimizations.py` | 保留 Q/K 投影 BF16；V/MLP 和输出投影使用行缩放 E4M3 |
| `sage2_adapter.py`, `sage2_model_5090.py` | 5 个 video early + 28 个 joint block 接入 Sage2；INT8 per-thread QK / FP8 PV / FP32 累加，K/V smooth 均开启 |
| `sage2-current-stream.patch` | Sage CUDA kernel 使用当前 stream，支持编译路径的 CUDA Graph；不改变 attention 数学 |
| `lossless_kquant.py` | 融合 K 中心化与 INT8 quant，显式保留 BF16 舍入 |
| `lossless_codegen_pinned.py`, `lossless_norm_fixed.py` | 保留算术与归约配置，Norm/RoPE 直接写 BF16/NHD；V 引用既有投影输出 |
| `lossless_vtranspose.py` | 64×64 tiled V 转置/padding/permutation；继续使用原 V smooth/quant |
| `lossless_constants.py`, `lossless_runtime.py` | 缓存相同依赖下的 timestep modulation、solver 系数、噪声与位置 IDs |
| `lossless_launch_pin.py`, `lossless_launch_manifest.json` | 固定通过 gate 的 46 个实际 hot launcher 配置 |
| `lossless_install.py` | 基线 bootstrap、启用优化并检查首次输出逐位一致 |
| `benchmark_lossless_selected.py` | 最终单独进程入口，记录源码和二进制哈希 |
| `benchmark_lossless_deploy.py` | 固定基线的重启验证、完整候选和交替 ABBA 测量 |
| `benchmark_5090.py`, `benchmark_sage2_5090.py`, `compare_5090.py` | 原计时、模型加载及质量比较 harness |

该 5090 路线使用官方 **SageAttention2**，不是 B200 的 VC-Attn kernel。
Sage 源码基点为 `thu-ml/SageAttention`
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`，仅附四文件 current-stream 补丁。
派生 K quant 文件保留上游 Apache-2.0 版权声明；许可证见仓库根目录
[`LICENSE`](../../LICENSE)。没有 vendor 整个 Sage 仓库。

被测 FLUX 源码为本仓库 `823db016f814d466be0b140434d71909181bcff0`；
原始 BF16 对照版本为 `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f`。
模型 revision 见 [`downloads.example.json`](downloads.example.json)。

## 已验证的环境和边界

Python 3.12；Torch 2.12.1+cu130；TorchVision 0.27.1+cu130；Triton 3.7.1；
SageAttention 2.2.0；NATTEN 0.21.7+torch2120cu130；Transformers 5.16.1；
NumPy 2.2.6；CUDA toolkit 13.0；单张 RTX 5090（SM120）。

这是固定模型/图形状的已测快照，不是通用编译器 pass。支持的长 attention 形状为
B1/N2720 与 B2/N3173、24 heads、head dim 128，覆盖 33 个 block。
launcher manifest 绑定完整生成代码 SHA256
`c051c5b0d49dcbcf2064b9c96e5aa5fdbe01a60eecea57873b037f7d3b145e45`。
源码路径、依赖版本或图结构变化可能触发 `baseline graph changed`。
遇到此错误应重新验证生成代码和质量，再生成新的 manifest；不要删除断言或
把新 SHA 填进去就当作通过验证。重新加载权重时需重新准备模型并清空 modulation cache。

固定 launcher 解决了本实验重启时三个 LayerNorm 核在 R0_BLOCK 512/1024
间变化导致的质量差异；未声称跨任意环境都逐位稳定。

## 在已有验证环境中复现

优先复用原有环境、模型和观测。三个 shell 入口只增加目录参数化：
`FLUX5090_ROOT` 指定运行目录，未设置时使用脚本所在目录。
该目录需要保留原来的结构：

```text
runtime/
  *.py, *.sh, sage2-current-stream.patch, lossless_launch_manifest.json
  venv/bin/python
  reference/src/flux_action/   # 被测优化源码
  downloads.json             # 从 example 填入本地模型路径
  hf-cache/                  # 已下载的模型与 encoders
  data/observation-{0,1}.npz
  data/observation-{0,1}.json # 每个包含 task 字符串
  data/ground_truth-{0,1}.npy
  results/original-v1/        # report.json + actions-{0,1}.npy
```

`data` 使用原 `flux_action.inference.offline.load_observation` 格式。
原始观测、模型权重、环境及内部日志不在此源码导出中。`downloads.json`
必须指向相同 checkpoint；比较器还要求 baseline/candidate 的配置、模型 revision
记录（含路径）与观测 SHA256 一致。

将本目录顶层的 `.py`、`.sh`、`.patch` 与 `lossless_launch_manifest.json`
同步到已有运行目录后：

```bash
export FLUX5090_ROOT=/path/to/existing-5090-runtime
bash "$FLUX5090_ROOT/run_selected_5090.sh" sync-check-001
# 等效：
bash "$FLUX5090_ROOT/run_selected_lossless_5090.sh" sync-check-002 all
```

使用新的输出名，已有目录会拒绝覆盖。入口先运行固定基线获得编译对象，再启用
全部优化并检查 bootstrap；计时及哈希写入 `results/<输出名>/`。
`compare_5090.py` 写出 `results/comparison-<输出名>.json`；需要检查其中
`all_quality_pass`，比较器本身不会因 gate 失败返回非零退出码。

完整 ABBA 复测还需要此前已验证的
`results/restart-diagnostic-v2/actions-{0,1}.npy` 作为跨进程逐位对照：

```bash
bash "$FLUX5090_ROOT/run_lossless_task.sh" benchmark_lossless_deploy.py abba-check-001
```

若需要重建 Sage，在上述同版本 CUDA/Python 环境下 checkout 指定 commit，
`git apply sage2-current-stream.patch`，然后使用
`TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 EXT_PARALLEL=1` 和该环境的
`python -m pip install --no-deps --no-build-isolation /path/to/SageAttention`。
需已有 Ninja 和 CUDA toolkit。此同步没有重新构建环境或重新运行 GPU 测试。

## 数据、网页与同步检查

- [`source-manifest.json`](source-manifest.json)：18 个原样源码/补丁文件的 SHA256，
  以及 3 个仅修改目录入口的 shell 文件的修改前后 SHA256。
- [`report/data/abba.json`](report/data/abba.json)：完整交替请求时间。
- [`report/data/quality.json`](report/data/quality.json)：原质量 gate。
- [`report/data/restart-verification.json`](report/data/restart-verification.json)：
  独立进程输出文件哈希与逐位验证。
- [`report/`](report/)：已发布 HF Space 的 13 个原样静态文件；
  [在线报告](https://huggingface.co/spaces/hp-l33/flux3-action-rtx5090-report)。

本次打包仅验证源文件哈希、Python/Bash 语法、Sage 补丁、引用完整性和公开文件
范围。核心 Python、数值阈值及 launcher manifest 与已测版本字节一致。
没有重新计时，也没有将目录参数化视为跨环境的 GPU 验证。
