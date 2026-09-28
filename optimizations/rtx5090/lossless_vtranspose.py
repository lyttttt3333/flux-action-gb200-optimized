"""Pure V layout transformation; Sage's V mean/scale/quant remains unchanged."""
import torch
import triton
import triton.language as tl
from sageattention import _fused

@triton.jit
def transpose(V,O, N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,P:tl.constexpr,
              SB:tl.constexpr,SN:tl.constexpr,SH:tl.constexpr,
              BN:tl.constexpr,BD:tl.constexpr):
    p=tl.program_id(0)*BN+tl.arange(0,BN)
    d=tl.program_id(1)*BD+tl.arange(0,BD)
    h=tl.program_id(2)%H;b=tl.program_id(2)//H
    src=(p//16)*16+p%2+((p//4)%4)*2+((p//2)%2)*8
    val=tl.load(V+b*SB+src[:,None]*SN+h*SH+d[None,:],(src[:,None]<N)&(d[None,:]<D),other=0)
    tl.store(O+b*D*H*P+d[None,:]*H*P+h*P+p[:,None],val,(p[:,None]<P)&(d[None,:]<D))

def permute(v,*,bn=64,bd=64,warps=4):
    b,n,h,d=v.shape;p=triton.cdiv(n,64)*64
    out=torch.empty((b,d,h,p),device=v.device,dtype=v.dtype)
    transpose[(triton.cdiv(p,bn),triton.cdiv(d,bd),b*h)](v,out,n,h,d,p,*v.stride()[:3],bn,bd,num_warps=warps)
    return out

def per_channel_fp8(v,tensor_layout='NHD',scale_max=448.,smooth_v=True):
    assert tensor_layout=='NHD' and smooth_v
    b,n,h,d=v.shape
    vt=permute(v)
    result=torch.empty(vt.shape,device=v.device,dtype=torch.float8_e4m3fn)
    scale=torch.empty((b,h,d),device=v.device,dtype=torch.float32);mean=torch.empty_like(scale)
    _fused.mean_scale_fuse_quant_cuda(vt,result,mean,scale,n,scale_max,0)
    return result,scale,mean
