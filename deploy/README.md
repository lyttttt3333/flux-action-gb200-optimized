# FLUX 3 Action deployment

The deployment uses the released DROID BF16 policy and the official OpenPI-compatible
RoboLab WebSocket server. Model and encoder revisions are pinned by the downloaded policy
package; jobs run with Hugging Face offline mode enabled. The long-running service enables
the compiled DiT/VAE path and uses cuDNN SDPA for the long video and joint-attention streams.
Positive and negative classifier-free-guidance branches with compatible text buckets are evaluated
as one batch, while incompatible buckets automatically retain the serial fallback.
Its first warm-up compiles the static request shapes and can take roughly three minutes on GB200.

The latency deployment uses all four GB200s as two tensor-parallel replicas. Each TP=2 replica
runs one classifier-free-guidance branch, and the guided predictions are reduced before every
UniPC update. Image/state payloads and the steady prompt-control path use NCCL tensors; Gloo is
used only when the prompt changes. The four-step UniPC arithmetic is compiled and fused without
changing its schedule or the four DiT evaluations.

Run and validate one recorded observation:

```bash
sbatch -A <slurm-account> deploy/flux3_action_smoke.sbatch
```

After the smoke test succeeds, start the four-GPU service for 24 hours:

```bash
sbatch -A <slurm-account> deploy/flux3_action_distributed.sbatch
```

`deploy/flux3_action_service.sbatch` remains the slower single-GPU fallback.

For the fastest measured single-GB200 mode, start:

```bash
sbatch -A <slurm-account> deploy/flux3_action_singlecard_fast.sbatch
```

This selects the released rowwise-FP8 checkpoint, tuned SM100 Quack GEMMs,
and three UniPC evaluations. On the recorded DROID request it measured
133.63 ms mean and 134.99 ms P95 over 30 steady-state requests. It is an
approximate latency tier: the checkpoint is FP8 and reducing four NFE to
three changes the solver result. Install the Blackwell kernel dependency with
`uv sync --extra encoders --extra serve --extra blackwell`. The cluster QOS
allocates the whole four-GPU node, but the script masks it to one visible GPU;
the model and measured latency are strictly single-card.

The service writes its `ws://node:port` address to
`outputs/flux3-action-service-<job-id>.endpoint`. It accepts the OpenPI request schema
documented in `docs/setup.md`; the service predicts action chunks but never sends commands
to robot hardware by itself.

Benchmark the steady state with the same recorded observation on one persistent connection:

```bash
endpoint=$(cat outputs/flux3-action-service-<job-id>.endpoint)
.venv/bin/python deploy/benchmark_service.py \
  --endpoint "$endpoint" \
  --observation outputs/public-droid/observation.npz \
  --prompt 'Pour the contents of the yellow cup into the pink bowl' \
  --warmup 5 --iterations 30 \
  --output benchmarks/droid-service-<job-id>.json
```

The warm-up count matters: the first real prompt may use a different text bucket than the service's
synthetic compile warm-up and can trigger one additional graph capture. See `benchmarks/RESULTS.md`
for the measured GB200 baseline and optimized results.

The Slurm launchers resolve the repository from their own location. Override
`FLUX_ACTION_REPO`, `FLUX_ACTION_CHECKPOINT`, `FLUX_ACTION_CACHE_ROOT`, or
`FLUX_ACTION_ENV_SETUP` when your checkout, weights, cache, or environment setup script lives
elsewhere. The RoboLab launcher additionally requires `ISAAC_IMAGE` and accepts
`ROBOLAB_REPO`/`AGENT_DEPLOY_ROOT` overrides.
