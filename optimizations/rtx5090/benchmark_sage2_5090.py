"""Sage2 candidate with unchanged model loading/timing/quality harness."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import sageattention
from flux_action.models.transformer_inf_bf16 import BF16InferenceDiT
from sage2_model_5090 import install

root = Path(sys.argv[1])
output, mode = sys.argv[2:4]
layout = sys.argv[4] if len(sys.argv)>4 else 'NHD'
original = BF16InferenceDiT.from_state_dict.__func__


def from_state_dict(cls, state_dict):
    model = original(cls, state_dict)
    execution = install(model, mode, layout)
    execution['source_sha256'] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ['sage2_model_5090.py','sage2_adapter.py','sm120_optimizations.py',
                     'sage2-current-stream.patch','benchmark_sage2_5090.py','benchmark_5090.py']}
    package = Path(sageattention.__file__).parent
    execution['sage_binary_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in package.glob('*.so')}
    (root/'results'/output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    print(json.dumps({'adapter_installed':execution}),flush=True)
    return model


BF16InferenceDiT.from_state_dict = classmethod(from_state_dict)
sys.argv = [str(root/'benchmark_5090.py'),str(root),'--mode','optimized',
            '--output',output,'--warmup','5','--repeats','30']
runpy.run_path(str(root/'benchmark_5090.py'),run_name='__main__')
path = root/'results'/output/'report.json'
report = json.loads(path.read_text())
report['execution'] = json.loads((path.parent/'execution.json').read_text())
path.write_text(json.dumps(report,indent=2)+'\n')
