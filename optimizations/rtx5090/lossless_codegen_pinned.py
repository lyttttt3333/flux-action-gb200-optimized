"""Conservative post-codegen store/alias optimization for the pinned 5090 graph.

Arithmetic text is preserved from Inductor; only output dtype/layout and the
now-redundant copies change. Unexpected graph structures raise an error.
"""
import ast
import hashlib
import re
from pathlib import Path

def transform(source, qk=True, value=True):
    tree=ast.parse(source)
    kernels={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='triton':
            kernels[node.targets[0].id]=ast.literal_eval(node.value.args[1])
    norms={name:body for name,body in kernels.items() if 'add_cat_mean_mul_pow_rsqrt_select' in name and "'out_ptr2': '*fp32'" in body}
    assert len(norms)==2, ('unexpected norm families', list(norms))
    original=source
    countq=countv=0
    for name,body in norms.items():
        n=3173 if 'xnumel = 152304' in body else 2720
        b=2 if n==3173 else 1
        shape=f'({b}, 24, {n}, 128)'
        strides=f'({n*3072}, 128, 3072, 1)'
        if qk:
            changed=body.replace("'out_ptr2': '*fp32'", "'out_ptr2': '*bf16'").replace("'out_ptr3': '*fp32'", "'out_ptr3': '*bf16'")
            changed,c=re.subn(r'tl.store\((out_ptr[23]) \+ \([^\n]+?\), (tmp(?:69|103)), r0_mask & xmask\)',r'tl.store(\1 + (r0_2 + 128*x0 + 3072*x1), \2, r0_mask & xmask)',changed)
            assert c==2
            source=source.replace(body,changed,1)
        pattern=r'^        '+re.escape(name)+r'\.run\(([^\n]+)\)$'
        calls=list(re.finditer(pattern,source,re.M))
        assert len(calls)==(28 if n==3173 else 5),len(calls)
        for call in calls:
            args=call.group(1).split(', ')
            qtmp,ktmp=args[5:7]
            if qk:
                for tmp in (qtmp,ktmp):
                    allocation=rf'^        {tmp} = ([^\n]+)$'
                    old=re.search(allocation,source,re.M);assert old,tmp
                    tail=';'+old.group(1).split(';',1)[1] if ';' in old.group(1) else ''
                    source,c=re.subn(allocation,f'        {tmp} = empty_strided_cuda(({b}, 24, {n}, 64, 2), ({n*3072}, 128, 3072, 2, 1), torch.bfloat16)'+tail,source,flags=re.M)
                    assert c==1,(tmp,c)
                    cp=rf'^        (triton_poi_[\w]+)\.run\({tmp}, (buf\d+), \d+, stream=raw_stream0\)$'
                    cm=re.search(cp,source,re.M);assert cm,tmp
                    dest=cm.group(2)
                    assignment=rf'^        {dest} = ([^\n]+)$'
                    am=re.search(assignment,source,re.M);assert am,dest
                    suffix=';'+am.group(1).split(';',1)[1] if ';' in am.group(1) else ''
                    source=source[:am.start()]+f'        {dest} = reinterpret_tensor({tmp}, {shape}, {strides}, 0)'+suffix+source[am.end():]
                    source,c=re.subn(cp,'        # lossless: BF16/NHD is written by Norm/RoPE.',source,flags=re.M);assert c==1
                countq+=1
    if value:
        vnames=[name for name,body in kernels.items() if name.startswith('triton_poi_') and 'attention_cat_split' in name and "'in_ptr1': '*bf16'" in body]
        assert len(vnames)==2,vnames
        for name in vnames:
            pattern=r'^        '+re.escape(name)+r'\.run\((buf\d+), (buf\d+), (buf\d+), (\d+), stream=raw_stream0\)$'
            for match in list(re.finditer(pattern,source,re.M)):
                qbuf,vbuf,dest,elements=match.groups()
                b,n=(2,3173) if int(elements)==19494912 else (1,2720)
                alloc=rf'^        {dest} = ([^\n]+)$'
                old=re.search(alloc,source,re.M);assert old,dest
                tail=';'+old.group(1).split(';',1)[1] if ';' in old.group(1) else ''
                repl=f'        {dest} = reinterpret_tensor({vbuf}, ({b}, 24, {n}, 128), ({n*21504}, 128, 21504, 1), 0)'
                source,c=re.subn(alloc,repl+tail,source,flags=re.M);assert c==1,(dest,c)
                countv+=1
            source=re.sub(pattern,'        # lossless: V aliases its existing BF16 projection.',source,flags=re.M)
    assert countq==(33 if qk else 0) and countv==(33 if value else 0),(countq,countv)
    ast.parse(source)
    return source,{'qk_blocks':countq,'v_blocks':countv,'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(source.encode()).hexdigest()}

class FixedNorm:
    def __init__(self, original, label):
        from lossless_norm_fixed import opt_early,opt_joint
        launchers=original.launchers
        assert len(launchers)==1, 'baseline kernel must have selected its launch config'
        config=launchers[0].config
        self.kwargs=dict(config.kwargs)
        self.num_warps=config.num_warps;self.num_stages=config.num_stages
        assert self.kwargs['XBLOCK']==2 and self.kwargs['R0_BLOCK'] in (64,128)
        assert self.num_warps==2
        self.kernel=opt_joint if label=='joint' else opt_early
    def run(self,*args,stream):
        import torch,triton
        assert torch.cuda.current_stream().cuda_stream==stream
        self.kernel[(triton.cdiv(args[-2],self.kwargs['XBLOCK']),)](
            *args,**self.kwargs,num_warps=self.num_warps,num_stages=self.num_stages,enable_fp_fusion=True)

def _kernels(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='triton':
            result[node.targets[0].id]=ast.literal_eval(node.value.args[1])
    return result

def install_loader(output_dir):
    from torch._inductor.codecache import PyCodeCache
    baseline={};norms={}
    for module in PyCodeCache.modules:
        path=getattr(module,'__file__',None)
        if not path:continue
        source=Path(path).read_text()
        if 'torch.ops.flux5090_sage2.attention.default' not in source or 'xnumel = 152304' not in source:continue
        for name,body in _kernels(source).items():
            obj=getattr(module,name)
            baseline[hashlib.sha256(body.encode()).hexdigest()]=obj
            if 'add_cat_mean_mul_pow_rsqrt_select' in name:
                label='joint' if 'xnumel = 152304' in body else 'early'
                norms[label]=FixedNorm(obj,label)
    assert len(norms)==2, ('baseline norm kernels not found',list(norms))
    original=PyCodeCache.load_by_key_path.__func__;seen={}
    def load(cls,key,path,linemap=None,attrs=None):
        source=Path(path).read_text()
        if 'torch.ops.flux5090_lossless.attention.default' not in source or '# lossless: ' in source:
            return original(cls,key,path,linemap,attrs)
        qk=any(repr(s) in source for s in ('qk','qkv','kq','combined','all'))
        value=any(repr(s) in source for s in ('v','kv','qkv','combined','all'))
        if not (qk or value):return original(cls,key,path,linemap,attrs)
        if key not in seen:
            import json
            patched,meta=transform(source,qk=qk,value=value)
            newkey,newpath=cls.write(patched);seen[key]=(newkey,newpath)
            out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
            (out/f'{key}-before.py').write_text(source);(out/f'{key}-after.py').write_text(patched)
            meta['norm_configs']={k:{**v.kwargs,'num_warps':v.num_warps,'num_stages':v.num_stages} for k,v in norms.items()}
            (out/f'{key}.json').write_text(json.dumps(meta,indent=2)+'\n')
        newkey,newpath=seen[key]
        module=original(cls,newkey,newpath,linemap,attrs)
        for name,body in _kernels(source).items():
            if qk and 'add_cat_mean_mul_pow_rsqrt_select' in name:
                setattr(module,name,norms['joint' if 'xnumel = 152304' in body else 'early'])
            else:
                digest=hashlib.sha256(body.encode()).hexdigest()
                if digest in baseline:setattr(module,name,baseline[digest])
        return module
    PyCodeCache.load_by_key_path=classmethod(load)
    return seen
