"""Same BF16 DROID checkpoint, observations, four steps and seed on one RTX5090."""
import argparse,importlib.metadata as metadata,json,statistics,subprocess,time
from pathlib import Path
import numpy as np
import torch
from flux_action.policy import FluxActionPolicy
from flux_action.models.transformer_inf_bf16 import BF16InferenceDiT
from flux_action.inference.precision import prepare_for_serving
from flux_action.inference.offline import load_observation
from flux_action.checkpoints.artifacts import sha256

pa=argparse.ArgumentParser()
pa.add_argument('root',type=Path);pa.add_argument('--mode',choices=['original','optimized'],required=True)
pa.add_argument('--output',required=True);pa.add_argument('--warmup',type=int,default=5);pa.add_argument('--repeats',type=int,default=30)
args=pa.parse_args();root=args.root
out=root/'results'/args.output;out.mkdir(parents=True,exist_ok=False)
def occupancy():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True).strip()
report={'mode':args.mode,'source_commit':'e2dd1d8dbc5977b54315d61f7548c63c043d6d4f' if args.mode=='original' else '823db016f814d466be0b140434d71909181bcff0',
        'timing_scope':'warm predict_action_chunk including image/VAE conditioning and all four video/action denoising steps; excludes checkpoint load, Qwen text encoding, first compilation and warmups',
        'warmup':args.warmup,'repeats':args.repeats,'cases':[], 'occupancy_start':occupancy(),
        'runtime':{p:metadata.version(p) for p in ['torch','torchvision','triton','natten','transformers','numpy']}}
def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
downloads=json.loads((root/'downloads.json').read_text())
checkpoint=Path(downloads['droid']['path'])
policy=FluxActionPolicy.from_pretrained(checkpoint,device='cpu')
policy.config.validate_inference()
report['policy_config']=policy.config.to_dict()
report['checkpoint_manifest']=json.loads((checkpoint/'manifest.json').read_text())
report['model_revisions']=downloads
tasks=[json.loads((root/f'data/observation-{i}.json').read_text())['task'] for i in [0,1]]
policy.set_text_encoder_offload(True)
with torch.inference_mode():
    for caption in [*tasks,'']:policy._context(caption,torch.device('cuda'))
torch.cuda.synchronize()
contexts=dict(policy._ctx_cache)
# Verified preparation from this task: pack on CPU so one 32GB card does not
# transiently retain both original and packed full DiT weights on the GPU.
packed=BF16InferenceDiT.from_state_dict(policy.dit.state_dict())
torch.nn.Module._apply(packed,lambda tensor:tensor.to(device='cuda'))
policy.dit=packed
policy.video_vae.module.to('cuda')
report['setup']=prepare_for_serving(policy,offload_text_encoder=True,compile_dit=args.mode=='optimized')
policy._ctx_cache.update(contexts)
torch.cuda.empty_cache();save()
with torch.inference_mode():
    for case_id,task in enumerate(tasks):
        path=root/f'data/observation-{case_id}.npz'
        batch=load_observation(path,policy.config,task,torch.device('cuda'))
        row={'id':case_id,'task':task,'observation_sha256':sha256(path),'warmup_seconds':[],'seconds':[],
             'occupancy_before':occupancy(),'repeat_max_abs':0.0}
        first=None
        row['first_warmup_dispatches']=[]
        saved_forward=policy.dit.forward_prepared
        def observe_forward(*pos,**kw):
            row['first_warmup_dispatches'].append({'batch_size':pos[0].batch_size,
                'shared_cfg_inputs':bool(kw.get('shared_cfg_inputs',False))})
            return saved_forward(*pos,**kw)
        policy.dit.forward_prepared=observe_forward
        for iteration in range(args.warmup+args.repeats):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();actions=policy.predict_action_chunk(batch);torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            if iteration==0:policy.dit.forward_prepared=saved_forward
            arr=actions.float().cpu().numpy()
            assert arr.shape==(1,32,8) and np.isfinite(arr).all()
            if first is None:first=arr.copy()
            row['repeat_max_abs']=max(row['repeat_max_abs'],float(np.abs(arr-first).max()))
            row['warmup_seconds' if iteration<args.warmup else 'seconds'].append(seconds)
            row['peak_allocated_bytes']=max(row.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated())
            print(json.dumps({'case':case_id,'iteration':iteration,'mode':args.mode,'seconds':seconds}),flush=True)
        np.save(out/f'actions-{case_id}.npy',arr)
        row['mean_seconds']=statistics.mean(row['seconds']);row['median_seconds']=statistics.median(row['seconds'])
        row['p95_seconds']=float(np.percentile(row['seconds'],95));row['occupancy_after']=occupancy()
        report['cases'].append(row);save()
report['status']='COMPLETE';report['occupancy_end']=occupancy();save()
