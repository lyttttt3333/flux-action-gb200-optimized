"""Exact-work elimination atop the selected Sage2/v3 policy."""
import types
import torch
from torch import nn
import sageattention.core as core
from lossless_kquant import per_thread_int8_fused

# A private function namespace avoids changing baseline Sage/global state.
_globals=dict(vars(core));_globals['per_thread_int8_triton']=per_thread_int8_fused
_fused=types.FunctionType(core.sageattn_qk_int8_pv_fp8_cuda.__code__,_globals,
                         'sage_with_fused_k',core.sageattn_qk_int8_pv_fp8_cuda.__defaults__)
_fused.__kwdefaults__=core.sageattn_qk_int8_pv_fp8_cuda.__kwdefaults__
from lossless_vtranspose import per_channel_fp8 as _per_channel_fp8
_all_globals=dict(_globals);_all_globals['per_channel_fp8']=_per_channel_fp8
_fused_all=types.FunctionType(core.sageattn_qk_int8_pv_fp8_cuda.__code__,_all_globals,
                             'sage_with_lossless_preparation',core.sageattn_qk_int8_pv_fp8_cuda.__defaults__)
_fused_all.__kwdefaults__=core.sageattn_qk_int8_pv_fp8_cuda.__kwdefaults__


@torch.library.custom_op('flux5090_lossless::attention',mutates_args=(),device_types='cuda')
def attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mode:str)->torch.Tensor:
    fn=_fused_all if mode=='all' else (_fused if mode in ('k','kv','kq','combined') else core.sageattn_qk_int8_pv_fp8_cuda)
    result=fn(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        tensor_layout='NHD',is_causal=False,qk_quant_gran='per_thread',
        pv_accum_dtype='fp32',smooth_k=True,smooth_v=True,return_lse=False)
    return result.flatten(2)

@attention.register_fake
def _fake(q,k,v,mode):
    return torch.empty((q.shape[0],q.shape[2],q.shape[1]*q.shape[3]),device=q.device,dtype=q.dtype)

class LosslessAttention(nn.Module):
    def __init__(self,mode):super().__init__();self.mode=mode
    def forward(self,q,k,v):return attention(q,k,v,self.mode)

def set_attention(model,mode):
    blocks=list(model.single_blocks)+list(model.content_mode_blocks['video'])
    for block in blocks:block.attention=LosslessAttention(mode)

class ExactStepCache:
    """Model-bound cache for immutable serving weights; clear on weight reload.

    Full vector/timestep equality prevents stale reuse after caption, schedule,
    batch, device or dtype changes. Observation-dependent tensors are not cached.
    """
    def __init__(self,model):
        self.model=model;self.original=model.prepare_steps;self.entry=None
        self.hits=0;self.misses=0
    def clear(self):self.entry=None
    def __call__(self,request,video_times,action_times):
        inputs=(request.vector_embedding,video_times,action_times)
        signature=(request.batch_size,request.dtype,request.device,tuple((x.shape,x.dtype,x.device) for x in inputs))
        if self.entry is not None:
            key,tensors,values=self.entry
            if signature==key and all(torch.equal(x,y) for x,y in zip(inputs,tensors)):
                self.hits+=1;return values
        values=self.original(request,video_times,action_times)
        self.entry=(signature,tuple(x.detach().clone() for x in inputs),values)
        self.misses+=1
        return values
