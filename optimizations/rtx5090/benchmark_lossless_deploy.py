"""Validate frozen baseline, complete lossless candidate and alternating E2E."""
import copy,hashlib,json,runpy,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
from lossless_launch_pin import install_launch_pin
root=Path(sys.argv[1]);name=sys.argv[2];out=root/'results'/name;out.mkdir(exist_ok=False)
pinned=install_launch_pin(out/'pin')
original_run=runpy.run_path;retained={}
def retaining(path,*args,**kwargs):
    result=original_run(path,*args,**kwargs)
    if Path(path).name=='benchmark_5090.py':retained.update(result)
    return result
runpy.run_path=retaining
sys.argv=[str(root/'benchmark_sage2_5090.py'),str(root),name+'-baseline','sage2_smooth_thread','NHD']
original_run(str(root/'benchmark_sage2_5090.py'),run_name='__main__');runpy.run_path=original_run
policy=retained['policy'];model=policy.dit
base=json.loads((root/'results'/(name+'-baseline')/'report.json').read_text())
golden={i:np.load(root/'results'/(name+'-baseline')/f'actions-{i}.npy') for i in (0,1)}
state={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('lossless_*.py')},'pinned_modules':len(pinned),'baseline_matches_recorded':{},'cases':[]}
for i in (0,1):state['baseline_matches_recorded'][str(i)]=bool(np.array_equal(golden[i],np.load(root/'results/restart-diagnostic-v2'/f'actions-{i}.npy')))
(out/'paired.json').write_text(json.dumps(state,indent=2)+'\n')
assert len(pinned)==1 and all(state['baseline_matches_recorded'].values()), 'frozen baseline failed restart identity'
print(json.dumps({'baseline_restart_exact':state['baseline_matches_recorded']}),flush=True)
from lossless_install import activate
original_methods=policy._sample_prepared,policy._sample;original_steps=model.prepare_steps
with torch.inference_mode():
    batch=retained['load_observation'](root/'data/observation-0.npz',policy.config,retained['tasks'][0],torch.device('cuda'))
    state['activation']=activate(policy,batch,mode='all',evidence_dir=out)
optimized_methods=policy._sample_prepared,policy._sample;optimized_steps=model.prepare_steps
candidate=root/'results'/(name+'-all');candidate.mkdir(exist_ok=False)
report=copy.deepcopy(base);report['cases']=[];report.pop('status',None);report['execution']['lossless']=state['activation']
with torch.inference_mode():
    for i,task in enumerate(retained['tasks']):
        batch=retained['load_observation'](root/f'data/observation-{i}.npz',policy.config,task,torch.device('cuda'))
        r=copy.deepcopy(base['cases'][i]);r['warmup_seconds']=[];r['seconds']=[];r['repeat_max_abs']=0.;first=None;equal=True
        for it in range(35):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            a=policy.predict_action_chunk(batch);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            arr=a.float().cpu().numpy();assert np.isfinite(arr).all()
            if first is None:first=arr.copy()
            equal=equal and bool(np.array_equal(arr,golden[i]));r['repeat_max_abs']=max(r['repeat_max_abs'],float(np.abs(arr-first).max()))
            r['warmup_seconds' if it<5 else 'seconds'].append(elapsed)
            r['peak_allocated_bytes']=max(r.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated())
            if it in (0,4,34):print(json.dumps({'candidate_case':i,'it':it,'seconds':elapsed,'equal':equal}),flush=True)
        r['median_seconds']=statistics.median(r['seconds']);r['mean_seconds']=statistics.mean(r['seconds']);r['p95_seconds']=float(np.percentile(r['seconds'],95))
        report['cases'].append(r);np.save(candidate/f'actions-{i}.npy',arr)
        state['cases'].append({'id':i,'all_equal':equal});assert equal
report['status']='COMPLETE';(candidate/'report.json').write_text(json.dumps(report,indent=2)+'\n')
from sage2_model_5090 import SageAttention
from lossless_runtime import set_attention
def select(enabled):
    policy._sample_prepared,policy._sample=optimized_methods if enabled else original_methods
    model.prepare_steps=optimized_steps if enabled else original_steps
    if enabled:set_attention(model,'all')
    else:
        for block in list(model.single_blocks)+list(model.content_mode_blocks['video']):block.attention=SageAttention('sage2_smooth_thread','NHD')
ab={'schedule':'5 warmups/path/input; 15 ABBA groups =30 timed requests/path/input','cases':[]}
with torch.inference_mode():
    for i,task in enumerate(retained['tasks']):
        batch=retained['load_observation'](root/f'data/observation-{i}.npz',policy.config,task,torch.device('cuda'))
        for enabled in (False,True):
            select(enabled)
            for _ in range(5):policy.predict_action_chunk(batch)
        rows={False:[],True:[]};equal=True
        for group in range(15):
            for enabled in [False,True,True,False]:
                select(enabled);torch.cuda.synchronize();start=time.perf_counter()
                a=policy.predict_action_chunk(batch);torch.cuda.synchronize();rows[enabled].append(time.perf_counter()-start)
                equal=equal and bool(np.array_equal(a.float().cpu().numpy(),golden[i]))
        r={'id':i,'all_equal':equal,'baseline_seconds':rows[False],'candidate_seconds':rows[True],
           'baseline_median':statistics.median(rows[False]),'candidate_median':statistics.median(rows[True])}
        r['speedup']=r['baseline_median']/r['candidate_median'];ab['cases'].append(r)
        (out/'abba.json').write_text(json.dumps(ab,indent=2)+'\n');assert equal
        print(json.dumps({'abba':i,'baseline':r['baseline_median'],'candidate':r['candidate_median'],'speedup':r['speedup'],'exact':equal}),flush=True)
for target in (name+'-baseline',name+'-all'):
    sys.argv=[str(root/'compare_5090.py'),str(root),'--candidate',target]
    original_run(str(root/'compare_5090.py'),run_name='__main__')
    state[target+'_quality']=json.loads((root/'results'/('comparison-'+target+'.json')).read_text())['all_quality_pass']
state['abba']=ab;state['status']='COMPLETE';(out/'paired.json').write_text(json.dumps(state,indent=2)+'\n')
assert state[name+'-baseline_quality'] and state[name+'-all_quality']
