"""Freeze the observed hot-graph launchers; never alter kernel arithmetic."""
import hashlib,json
from pathlib import Path

def install_launch_pin(evidence_dir,manifest_path=None):
    from torch._inductor.codecache import PyCodeCache
    import triton
    manifest_path=Path(manifest_path or Path(__file__).with_name('lossless_launch_manifest.json'))
    manifest=json.loads(manifest_path.read_text())
    original=PyCodeCache.load_by_key_path.__func__;seen={}
    out=Path(evidence_dir)
    def config(c):return {'kwargs':dict(c.kwargs),'num_warps':c.num_warps,'num_stages':c.num_stages}
    def load(cls,key,path,linemap=None,attrs=None):
        source=Path(path).read_text()
        module=original(cls,key,path,linemap,attrs)
        if 'torch.ops.flux5090_sage2.attention.default' not in source:return module
        digest=hashlib.sha256(source.encode()).hexdigest()
        assert digest==manifest['module_sha256'], 'baseline graph changed; launch manifest needs validation'
        applied={}
        for name,expected in manifest['kernels'].items():
            obj=getattr(module,name);obj.precompile()
            options=[x for x in obj.launchers if config(x.config)==expected]
            if not options:
                if obj.fn.fn is None:obj.fn=obj._reload_kernel().fn
                compiled=obj._precompile_config(triton.Config(**expected))
                options=[compiled.make_launcher()]
            assert len(options)==1,(name,len(options))
            obj.launchers=options
            obj.inductor_meta['coordinate_descent_tuning']=False
            obj.inductor_meta['combo_tuning_groups']=None
            applied[name]=config(options[0].config)
        assert applied==manifest['kernels']
        seen[key]={'module_sha256':digest,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'actual':applied}
        out.mkdir(parents=True,exist_ok=True)
        (out/'launch_pin.json').write_text(json.dumps(seen,indent=2)+'\n')
        return module
    PyCodeCache.load_by_key_path=classmethod(load)
    return seen
