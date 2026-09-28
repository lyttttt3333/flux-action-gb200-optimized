"""Compare fixed-workload original and optimized BF16 FLUX Action runs."""
import argparse,json
from pathlib import Path
import numpy as np

def metrics(pred,reference):
    delta=pred.astype(np.float64)-reference.astype(np.float64)
    return {"joint_mae_rad":float(np.abs(delta[...,:7]).mean()),
            "joint_max_abs_rad":float(np.abs(delta[...,:7]).max()),
            "gripper_mae":float(np.abs(delta[...,7]).mean()),
            "gripper_max_abs":float(np.abs(delta[...,7]).max())}
p=argparse.ArgumentParser();p.add_argument('root',type=Path)
p.add_argument('--baseline',default='original-v1');p.add_argument('--candidate',default='optimized-v1');a=p.parse_args()
paths=[a.root/'results'/s for s in (a.baseline,a.candidate)]
reports=[json.loads((s/'report.json').read_text()) for s in paths]
b,c=reports
assert all(r['status']=='COMPLETE' for r in reports)
assert b['policy_config']==c['policy_config']
assert b['checkpoint_manifest']==c['checkpoint_manifest']
assert b['model_revisions']==c['model_revisions']
result={'baseline':a.baseline,'candidate':a.candidate,'cases':[],
        'scope':b['timing_scope'],'quality_thresholds':{'joint_mae_rad':.02,'joint_max_abs_rad':.05,'gripper_max_abs':.05,'recorded_joint_mae_degradation_rad':.02}}
for rb,rc in zip(b['cases'],c['cases'],strict=True):
    i=rb['id'];assert i==rc['id'] and rb['observation_sha256']==rc['observation_sha256']
    actions=[np.load(s/f'actions-{i}.npy') for s in paths]
    truth=np.load(a.root/f'data/ground_truth-{i}.npy')[None]
    diff=metrics(actions[1],actions[0]);against=[metrics(x,truth) for x in actions]
    result['cases'].append({'id':i,'original_seconds':rb['median_seconds'],'optimized_seconds':rc['median_seconds'],
        'speedup':rb['median_seconds']/rc['median_seconds'],'latency_reduction_pct':100*(1-rc['median_seconds']/rb['median_seconds']),
        'vs_original':diff,'against_recorded':dict(zip(('original','optimized'),against)),
        'quality_screen_pass':diff['joint_mae_rad']<=.02 and diff['joint_max_abs_rad']<=.05 and diff['gripper_max_abs']<=.05 and against[1]['joint_mae_rad']<=against[0]['joint_mae_rad']+.02,
        'repeat_max_abs':{'original':rb['repeat_max_abs'],'optimized':rc['repeat_max_abs']},
        'peak_allocated_bytes':{'original':rb['peak_allocated_bytes'],'optimized':rc['peak_allocated_bytes']}})
assert len(result['cases'])==2
result['all_quality_pass']=all(x['quality_screen_pass'] for x in result['cases'])
result['all_faster']=all(x['speedup']>1 for x in result['cases'])
target=a.root/'results'/f'comparison-{a.candidate}.json';target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
