"""Use the original loader/timing harness with one-time lossless bootstrap."""
import hashlib,json,runpy,sys
from pathlib import Path
from flux_action.policy import FluxActionPolicy
root=Path(sys.argv[1]);output=sys.argv[2];mode=sys.argv[3]
from lossless_launch_pin import install_launch_pin
pinned=install_launch_pin(root/'results'/output/'pin')
original=FluxActionPolicy.predict_action_chunk
execution={};initialized=False

def predict(self,batch):
    global initialized
    if not initialized:
        from lossless_install import activate
        # Temporarily restore the method to avoid recursion during the bootstrap.
        FluxActionPolicy.predict_action_chunk=original
        execution.update(activate(self,batch,mode=mode,evidence_dir=root/'results'/output))
        assert len(pinned)==1, 'expected pinned baseline graph'
        execution['launch_pin']=dict(pinned)
        initialized=True
        execution['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in [
            'lossless_install.py','lossless_runtime.py','lossless_kquant.py','lossless_codegen_pinned.py',
            'lossless_norm_fixed.py','lossless_constants.py','lossless_vtranspose.py','lossless_launch_pin.py',
            'lossless_launch_manifest.json','benchmark_lossless_selected.py']}
        (root/'results'/output/'lossless_execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    return original(self,batch)
FluxActionPolicy.predict_action_chunk=predict
sys.argv=[str(root/'benchmark_sage2_5090.py'),str(root),output,'sage2_smooth_thread','NHD']
runpy.run_path(str(root/'benchmark_sage2_5090.py'),run_name='__main__')
path=root/'results'/output/'report.json';report=json.loads(path.read_text());report['execution']['lossless']=execution
path.write_text(json.dumps(report,indent=2)+'\n')
