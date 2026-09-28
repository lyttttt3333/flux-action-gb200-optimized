"""Long-stream Sage2 integration atop the unchanged passing v3 projections."""
from torch import nn
from flux_action.models import transformer_inf_bf16 as base
from sm120_optimizations import install as install_projections
from sage2_adapter import sage_attention, MODES, SOURCE_COMMIT


class SageAttention(nn.Module):
    def __init__(self, mode, layout):
        super().__init__()
        self.mode = mode
        self.layout = layout

    def forward(self, q, k, v):
        return sage_attention(q, k, v, self.mode, self.layout)


def install(model, mode='sage2_smooth_thread', layout='NHD'):
    assert mode in MODES and layout in ('NHD','HND')
    execution = install_projections(model, 'rowwise-fp8-qkbf16-v3')
    blocks = [(f'single_blocks.{i}', b) for i,b in enumerate(model.single_blocks)]
    blocks += [(f'content_mode_blocks.video.{i}', b)
               for i,b in enumerate(model.content_mode_blocks[base.VIDEO])]
    for _, block in blocks:
        block.attention = SageAttention(mode, layout)
    execution['attention'] = {
        'implementation': mode, 'upstream_commit': SOURCE_COMMIT,
        'source_patch': 'sage2-current-stream.patch; CUDA launch stream only',
        'qk': 'INT8 per_thread' if mode.endswith('_thread') else 'INT8 per_warp',
        'pv': 'FP8', 'pv_accum_dtype':MODES[mode][0],
        'smooth_k':True, 'smooth_v':MODES[mode][1], 'layout':layout,
        'blocks':[name for name,_ in blocks], 'output_dtype':'bfloat16',
        'torch_compile':'fullgraph=True, mode=reduce-overhead; opaque custom op includes all preparation',
    }
    return execution
