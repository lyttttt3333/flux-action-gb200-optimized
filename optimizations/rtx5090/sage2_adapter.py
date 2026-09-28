"""Official SageAttention2 at d1a57a5, isolated behind a CUDA-graph-safe op.

The upstream kernel math is unchanged; CUDA launches use the current stream.
Inputs are logical HND;
the op includes smoothing, quantization and output layout conversion.
"""
import torch
from sageattention import sageattn_qk_int8_pv_fp8_cuda

MODES = {
    'sage2pp': ('fp32+fp16', False),
    'sage2_fp32': ('fp32', False),
    'sage2_smooth': ('fp32', True),
    'sage2_smooth_thread': ('fp32', True),
}
SOURCE_COMMIT = 'd1a57a546c3d395b1ffcbeecc66d81db76f3b4b5'


@torch.library.custom_op('flux5090_sage2::attention', mutates_args=(), device_types='cuda')
def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mode: str, layout: str) -> torch.Tensor:
    accum, smooth_v = MODES[mode]
    if layout == 'NHD':
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    result = sageattn_qk_int8_pv_fp8_cuda(
        q, k, v, tensor_layout=layout, is_causal=False,
        qk_quant_gran='per_thread' if mode.endswith('_thread') else 'per_warp', pv_accum_dtype=accum,
        smooth_k=True, smooth_v=smooth_v, return_lse=False)
    if layout == 'HND':
        result = result.transpose(1, 2).contiguous()
    return result.flatten(2)


@sage_attention.register_fake
def _fake(q, k, v, mode, layout):
    return torch.empty((q.shape[0], q.shape[2], q.shape[1]*q.shape[3]),
                       dtype=q.dtype, device=q.device)
