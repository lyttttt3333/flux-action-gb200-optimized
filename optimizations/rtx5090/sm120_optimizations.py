"""Task-local SM120 projection adapter for the pinned BF16 prepared backend.

Row quantization follows reference transformer_inf_fp8r.py; block math follows
transformer_inf_bf16.py at 823db016. No SM100/Quack kernel is transplanted.
This is approximate W8A8 execution of the existing BF16 checkpoint, not the
separately released FP8 checkpoint. Attention and residual math remain BF16.
"""
import torch
import torch.nn.functional as F
from torch import nn
from flux_action.models import transformer_inf_bf16 as base


class RowwiseFP8Linear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        assert weight.dtype == torch.bfloat16 and weight.ndim == 2
        self.out_features, self.in_features = weight.shape
        rows = weight.detach().float()
        scale = (rows.abs().amax(dim=1, keepdim=True) / 448.).clamp_min(1e-12)
        self.register_buffer('weight_q', (rows / scale).clamp(-448., 448.).to(torch.float8_e4m3fn))
        self.register_buffer('weight_scale', scale.T.contiguous())

    def forward(self, x):
        flat = x.reshape(-1, self.in_features).float()
        scale = (flat.abs().amax(dim=1, keepdim=True) / 448.).clamp_min(1e-12)
        quantized = (flat / scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
        result = torch._scaled_mm(quantized, self.weight_q.T, scale,
                                  self.weight_scale, out_dtype=torch.bfloat16,
                                  use_fast_accum=True)
        return result.reshape(*x.shape[:-1], self.out_features)


class BF16QKInputProjection(nn.Module):
    """Keep score-forming Q/K projections BF16; quantize V and packed MLP-in."""
    def __init__(self, weight, hidden_size):
        super().__init__()
        self.qk_weight = nn.Parameter(weight[:2*hidden_size].detach().clone(), requires_grad=False)
        self.value_mlp = RowwiseFP8Linear(weight[2*hidden_size:])

    def forward(self, x):
        return torch.cat((F.linear(x, self.qk_weight), self.value_mlp(x)), dim=-1)


class FP8ModeBlock(base._ModeBlock):
    def forward(self, x, rope, modulation):
        shift, scale, gate = modulation
        modulated = (1 + scale) * self.pre_norm(x) + shift
        batch, length, _ = modulated.shape
        q, k, v, mlp = self.input_projection(modulated).split(
            (self.hidden_size, self.hidden_size, self.hidden_size, 2*self.mlp_hidden_dim), dim=-1)
        q = q.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        q, k = self.norm(q, k, v)
        q, k = base._apply_rope(q, k, rope)
        output = self.attn_out(self.attention(q, k, v)) + self.mlp_out(self.mlp_act(mlp))
        return x + gate * output


class FP8JointBlock(base._JointBlock):
    def forward(self, sequence, rope, lengths, modulations):
        modulated = base._modulate_segments(self.pre_norm(sequence), lengths, modulations)
        batch, length, _ = modulated.shape
        q, k, v, mlp = self.input_projection(modulated).split(
            (self.hidden_size, self.hidden_size, self.hidden_size, 2*self.mlp_hidden_dim), dim=-1)
        q = q.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        q, k = self.norm(q, k, v)
        q, k = base._apply_rope(q, k, rope)
        output = self.attn_out(self.attention(q, k, v)) + self.mlp_out(self.mlp_act(mlp))
        return sequence + base._gate_segments(output, lengths, modulations)


@torch.no_grad()
def install(model, variant='rowwise-fp8-qkbf16-v3'):
    assert model._packed and model._compiled_hot is None
    assert variant in ('rowwise-fp8-hot-v2', 'rowwise-fp8-qkbf16-v3')
    converted = []
    blocks = [(f'single_blocks.{i}', b, FP8JointBlock) for i, b in enumerate(model.single_blocks)]
    blocks += [(f'content_mode_blocks.video.{i}', b, FP8ModeBlock)
               for i, b in enumerate(model.content_mode_blocks[base.VIDEO])]
    for name, block, cls in blocks:
        assert block._tensor_parallel_group is None
        block.input_projection = (BF16QKInputProjection(block.in_proj_weight, block.hidden_size)
                                  if variant == 'rowwise-fp8-qkbf16-v3'
                                  else RowwiseFP8Linear(block.in_proj_weight))
        block.in_proj_weight = None
        block.attn_out = RowwiseFP8Linear(block.attn_out.weight)
        block.mlp_out = RowwiseFP8Linear(block.mlp_out.weight)
        block.__class__ = cls
        converted.append(name)
    model.eval()
    return {'variant': variant, 'quantization': 'per-token E4M3 activations / per-output-row E4M3 weights',
            'qk_projection_dtype': 'bfloat16' if variant == 'rowwise-fp8-qkbf16-v3' else 'float8_e4m3fn',
            'source_weights': 'same pinned BF16 checkpoint, quantized once at load',
            'converted_blocks': converted, 'sampling_changed': False,
            'attention_dtype': 'bfloat16', 'sampler_dtype': 'float32'}
