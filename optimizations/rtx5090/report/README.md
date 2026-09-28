---
title: FLUX.3 Action · RTX 5090
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: RTX 5090 优化报告：约 2.05× 加速、质量 gate 与可下载实测证据
---

# FLUX.3 Action · RTX 5090 optimization report

中文交互报告，记录 2026-09-28 已验证的单卡 RTX 5090 结果。

- 原始 BF16 warm `predict_action_chunk`：1396.406 / 1408.608 ms。
- 最终 ABBA median：684.603 / 685.827 ms，约 2.04–2.05×。
- 本轮相对固定 Sage2 混合精度基线的无损优化：延迟额外下降 5.26–5.29%。
- 120 次 ABBA 请求输出逐位一致；两个固定离线输入通过原质量 gate。

总路线包含 FP8 投影与 SageAttention2 近似。仅最后一轮搬运融合、缓存和固定配置优化相对既有混合精度基线无新增近似。历史阶段来自独立进程，不能当作严格单项消融。

Timing includes image/VAE conditioning and four video/action denoising steps. Checkpoint loading, Qwen text encoding, compilation and warmup are excluded. This is an offline numerical screen, not robot success-rate evidence.

Open `data/flux3-5090-public-evidence.zip` for selected numerical measurements, source hashes and output-hash verification. The site has no backend, dependencies, analytics, model weights or credentials.

Reference implementation: https://github.com/lyttttt3333/flux-action-gb200-optimized (pinned 823db016f814d466be0b140434d71909181bcff0).

Visual reference: https://transactions-layout-photographs-paperbacks.trycloudflare.com/ . Its hardware and simulation results are not used as this report's measurements.
