# Recorded DROID steady-state results on GB200

Input: `outputs/public-droid/observation.npz`, task `Pour the contents of the yellow cup into the
pink bowl`, and one persistent WebSocket connection. The baseline uses 5 discarded warm-up
requests and 30 measured requests; the final deployment was rechecked with 10 warm-ups and 300
measured requests. Each response contains 32 actions at 15 Hz (2.133 seconds of control trajectory).

| Deployment | Server mean | Server p50 | Server p95 | RTT mean | Chunks/s |
|---|---:|---:|---:|---:|---:|
| BF16 eager, Torch SDPA | 527.93 ms | 527.93 ms | 534.29 ms | 532.04 ms | 1.89 |
| BF16 compiled, hybrid cuDNN SDPA | 227.24 ms | 227.27 ms | 230.86 ms | 230.83 ms | 4.40 |
| BF16 compiled, hybrid cuDNN SDPA, batched CFG | 213.61 ms | 213.22 ms | 216.44 ms | 217.44 ms | 4.68 |
| BF16, 2×TP2 parallel CFG, fused UniPC | **83.73 ms** | **83.51 ms** | **85.72 ms** | **87.22 ms** | **11.94** |

The final four-GPU deployment is 6.31x faster in server inference and 6.10x faster end to end than
the original BF16 service. Its RTT mean is 87.22 ms and RTT p95 is 89.27 ms, both below the 90 ms
target. It generates the 2.133-second action chunk 25.47x faster than real time. Relative to the
previous batched-CFG deployment, server latency falls another 60.8%.

Direct-process variant measurements (same recorded input, 30 requests):

| Variant | Mean | p50 | Chunks/s | Semantics |
|---|---:|---:|---:|---|
| BF16 eager + hybrid cuDNN | 454.79 ms | 455.04 ms | 2.20 | Same base model |
| BF16 compiled + hybrid cuDNN | 228.22 ms | 228.06 ms | 4.38 | Same base model |
| BF16 compiled + hybrid cuDNN + batched CFG | 200.05 ms | 200.09 ms | 5.00 | Same base model and sampling |
| BF16 2×TP2 parallel CFG + fused UniPC | 81.77 ms | 81.70 ms | 12.23 | Same BF16 model, CFG, 4 NFE and schedule |
| FP8r compiled + hybrid cuDNN | 210.68 ms | 210.41 ms | 4.75 | Official rowwise-FP8 package |
| Guidance-distilled FP8r compiled | 109.29 ms | 109.18 ms | 9.15 | Official no-CFG model variant |
| Step-distilled FP8r compiled | 34.95 ms | 34.81 ms | 28.61 | Official one-step model variant |

The distilled results are model/sampling changes, not kernel-only speedups, and are therefore not
the default deployed endpoint. FP8r is also not promoted by default: it gives only another 7.7% over
compiled BF16 here while its same-seed action difference from BF16 eager is 0.0914 max absolute.
The guidance-distilled and step-distilled comparisons differ by 0.2396 and 0.1574 max absolute,
respectively; these figures are only numerical differences on one input, not task-success metrics.

The batched-CFG eager path was compared to the original serial-CFG path using the same seed and
produced exactly identical actions (max and mean absolute difference both zero). The compiled
path differs from eager BF16 by 0.0212 max absolute and 0.00438 mean absolute, consistent with
compiled floating-point reassociation rather than a model or sampler change.
The final distributed/fused path differs from the original single-GPU eager path by 0.01569 max
absolute and 0.00360 mean absolute on the same seed. A two-GPU batched-CFG TP block measured
0.959 ms and was not promoted because its projected end-to-end latency did not satisfy the
required 90 ms two-GPU limit.

Raw service JSON and Nsight reports are intentionally excluded from the repository because they
contain cluster hostnames and large profiler payloads. The table above records the validated
aggregate measurements.
