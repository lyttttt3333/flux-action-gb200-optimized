# Derived from FLUX Action 823db016, Apache-2.0; exact CPU constants only.
from functools import lru_cache
import types
import torch
from flux_action.inference import sampling as original_sampling
from flux_action import policy as original_policy
from flux_action.inference.sampling import (
    Samples, _bh_coefficients, _order2_coefficients, _unipc_step0,
    _unipc_step1, _unipc_step2, _unipc_final)
from collections.abc import Callable
from torch import Tensor
cosmos_unipc_schedule = lru_cache(maxsize=8)(original_sampling.cosmos_unipc_schedule)

@lru_cache(maxsize=8)
def solver_constants(shift, num_train_timesteps=1000):
    (sigmas, ticks) = cosmos_unipc_schedule(4, shift, num_train_timesteps)
    s = tuple((float(value) for value in sigmas))
    h01 = torch.log1p(-sigmas[1]) - torch.log(sigmas[1]) - (torch.log1p(-sigmas[0]) - torch.log(sigmas[0]))
    hphi01 = float(torch.expm1(-h01))
    h12 = torch.log1p(-sigmas[2]) - torch.log(sigmas[2]) - (torch.log1p(-sigmas[1]) - torch.log(sigmas[1]))
    hphi12 = float(torch.expm1(-h12))
    rk12 = float(-h01 / h12)
    correct_delta1 = -(1.0 - s[1]) * hphi01 * 0.5
    h23 = torch.log1p(-sigmas[3]) - torch.log(sigmas[3]) - (torch.log1p(-sigmas[2]) - torch.log(sigmas[2]))
    hphi23 = float(torch.expm1(-h23))
    (c2_last, c2_prev, c2_hist, c2_delta, c2_scale) = _order2_coefficients(sigmas, 2)
    rk23 = float(-h12 / h23)
    return sigmas, ticks, s, h01, hphi01, h12, hphi12, rk12, correct_delta1, h23, hphi23, c2_last, c2_prev, c2_hist, c2_delta, c2_scale, rk23

def cached_fused4(samples: Samples, predict_velocity: Callable[[Samples, Tensor], Samples], *, shift: float, num_train_timesteps: int=1000) -> Samples:
    (sigmas, ticks, s, h01, hphi01, h12, hphi12, rk12, correct_delta1, h23, hphi23, c2_last, c2_prev, c2_hist, c2_delta, c2_scale, rk23) = solver_constants(shift, num_train_timesteps)
    'Four-step UniPC with the unchanged denoiser calls and fused FP32 solver arithmetic.'
    keys = list(samples)
    if len(keys) != 2:
        raise ValueError('fused four-step sampler requires exactly two streams')
    (video_key, action_key) = keys
    (initial_v, initial_a) = (samples[video_key], samples[action_key])
    velocity = predict_velocity(samples, ticks[0])
    (xv, xa, x00v, x00a) = _unipc_step0(initial_v, initial_a, velocity[video_key], velocity[action_key], s[0], s[1] / s[0], -(1.0 - s[1]) * hphi01)
    velocity = predict_velocity({video_key: xv, action_key: xa}, ticks[1])
    (xv, xa, last_v, last_a, x01v, x01a) = _unipc_step1(xv, xa, velocity[video_key], velocity[action_key], initial_v, initial_a, x00v, x00a, s[1], s[1] / s[0], -(1.0 - s[1]) * hphi01, correct_delta1, s[2] / s[1], -(1.0 - s[2]) * hphi12, -(1.0 - s[2]) * hphi12 * 0.5, 1.0 / rk12)
    velocity = predict_velocity({video_key: xv, action_key: xa}, ticks[2])
    (xv, xa, last_v, last_a, x02v, x02a) = _unipc_step2(xv, xa, velocity[video_key], velocity[action_key], last_v, last_a, x00v, x00a, x01v, x01a, s[2], c2_last, c2_prev, c2_hist, c2_delta, c2_scale, s[3] / s[2], -(1.0 - s[3]) * hphi23, -(1.0 - s[3]) * hphi23 * 0.5, 1.0 / rk23)
    velocity = predict_velocity({video_key: xv, action_key: xa}, ticks[3])
    (xv, xa) = _unipc_final(xv, xa, velocity[video_key], velocity[action_key], s[3])
    return {video_key: xv, action_key: xa}

class NoiseCache:
    def __init__(self):self.entries={};self.hits=0;self.misses=0
    def get(self, seed, cfg, device, rich_history, n_pred):
        key=(seed, str(device), rich_history, n_pred, tuple(cfg.latent_hw),
             cfg.chunk_size,cfg.action_dim,cfg.n_obs_steps,cfg.fps,cfg.video_position_fps)
        if key in self.entries:
            self.hits+=1;return self.entries[key]
        self.misses+=1
        rng = torch.Generator().manual_seed(seed)
        video_noise = torch.randn(1, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw, generator=rng)
        (x_video, x_video_ids) = batched_prc_vid(video_noise, packing.video_time_ids(n_pred, packing.latent_frames(cfg.n_obs_steps) if rich_history else 1, 1, fps=cfg.video_position_fps if rich_history else cfg.fps))
        times = torch.arange(cfg.chunk_size).float()[None] / cfg.fps if rich_history else packing.default_action_times(1, cfg.chunk_size, cfg.fps)
        action_noise = torch.randn(1, cfg.action_dim, cfg.chunk_size, generator=rng)
        (x_action, x_action_ids) = batched_prc_action(action_noise, times_to_ids(times))

        values=tuple(x.to(device) for x in (x_video,x_video_ids,x_action,x_action_ids))
        if len(self.entries)>=4:self.entries.clear()
        self.entries[key]=values
        return values

def install_constants(policy, *, solver=True, noise=True):
    saved=(policy._sample_prepared,policy._sample)
    if solver:
        sampling=types.SimpleNamespace(**vars(original_sampling))
        sampling.cosmos_unipc_schedule=cosmos_unipc_schedule
        sampling.cosmos_unipc_order2_fused4=cached_fused4
        original=policy._sample_prepared.__func__
        namespace=dict(original.__globals__);namespace['sampling']=sampling
        cloned=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        cloned.__kwdefaults__=original.__kwdefaults__
        policy._sample_prepared=types.MethodType(cloned,policy)
    if noise:
        policy._lossless_noise_cache=NoiseCache()
        namespace=dict(vars(original_policy));namespace['NoiseCache']=NoiseCache
        cloned=types.FunctionType(sample_with_cached_noise.__code__,namespace)
        policy._sample=types.MethodType(cloned,policy)
    return saved

packing=original_policy.packing
batched_prc_vid=original_policy.batched_prc_vid
batched_prc_action=original_policy.batched_prc_action
times_to_ids=original_policy.times_to_ids

def sample_with_cached_noise(self, cond: dict[str, Tensor], caption: str, seed: int) -> Tensor:
    """Joint video + action denoising from pure noise -> ``(chunk, D)`` in model units / action_scale."""
    (cfg, m, mdt) = (self.config, self.modality, self.dtype_)
    (ak, ck) = (f'x_{m}', f'x_{m}_cond')
    device = cond['x_video_cond'].device
    rich_history = cfg.inference_profile == 'history'
    n_pred = packing.latent_frames(cfg.chunk_size) if rich_history else packing.latent_frames(cfg.window_frames) - 1
    x_video, x_video_ids, x_action, x_action_ids = self._lossless_noise_cache.get(seed, cfg, device, rich_history, n_pred)
    flow = {'x_video': x_video.to(device), ak: x_action.to(device)}
    fixed = {'x_video_ids': x_video_ids.to(device), f'{ak}_ids': x_action_ids.to(device), 'x_video_cond': cond['x_video_cond'].to(device, mdt), 'x_video_cond_ids': cond['x_video_cond_ids'].to(device), 'x_video_cond_timesteps': torch.zeros(1, cond['x_video_cond'].shape[1], device=device), ck: cond[ck].to(device, mdt), f'{ck}_ids': cond[f'{ck}_ids'].to(device), f'{ck}_timesteps': torch.zeros(1, cond[ck].shape[1], device=device), 'vector': torch.zeros(1, VEC_DIM, device=device, dtype=mdt)}
    guidance = {'x_video': cfg.guidance_scale, ak: cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action}
    ctx_c = self._context(caption, device)
    ctx_uc = self._context('', device) if any((g != 1.0 for g in guidance.values())) else None
    if self._inference_prepared:
        return self._sample_prepared(flow=flow, fixed=fixed, caption=caption, ctx_c=ctx_c, ctx_uc=ctx_uc, guidance=guidance)[ak][0].float() / cfg.action_scale
    assert isinstance(self.dit, JointSingleSeq), 'call prepare_inference before prediction'

    def predict(samples: dict[str, Tensor], t) -> dict[str, Tensor]:
        t = float(t) / 1000.0 if isinstance(t, Tensor) and (not t.is_floating_point()) else float(t)
        timesteps = {'x_video_timesteps': torch.full((1, samples['x_video'].shape[1]), t, device=device), f'{ak}_timesteps': torch.full((1, cfg.chunk_size), t, device=device)}
        model_in = {k: v.to(mdt) for (k, v) in samples.items()}
        if ctx_uc is None:
            (ctx, ctx_ids) = ctx_c
            pred = self.dit(**model_in, **fixed, **timesteps, ctx=ctx, ctx_ids=ctx_ids, timesteps_ctx=torch.zeros(ctx.shape[:2], device=device))
            pred = {k: pred[k] for k in model_in}
        else:
            pred = sampling.cfg_two_pass(self.dit, model_in, fixed, timesteps, ctx_uc, ctx_c, guidance)
        return {k: v.float() for (k, v) in pred.items()}
    if cfg.sampler == 'cosmos_unipc':
        out = sampling.cosmos_unipc_order2(flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift)
    else:
        out = sampling.euler(flow, predict, n_steps=cfg.num_inference_steps, alpha=cfg.sampler_shift)
    return out[ak][0].float() / cfg.action_scale
