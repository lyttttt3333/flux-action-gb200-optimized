"""Activate the measured 5090 lossless path after loading the existing policy."""
from pathlib import Path
import torch
from lossless_codegen_pinned import install_loader
from lossless_runtime import set_attention, ExactStepCache
from lossless_constants import install_constants

@torch.inference_mode()
def activate(policy, observation, *, mode, evidence_dir):
    # Obtain the actual baseline launch choices before replacing any computation.
    baseline=policy.predict_action_chunk(observation).detach().clone()
    torch.cuda.synchronize()
    configs=install_loader(Path(evidence_dir)/'codegen')
    set_attention(policy.dit,mode)
    policy.dit.prepare_steps=ExactStepCache(policy.dit)
    install_constants(policy,solver=True,noise=True)
    candidate=policy.predict_action_chunk(observation).detach().clone()
    torch.cuda.synchronize()
    exact=bool(torch.equal(baseline,candidate))
    if not exact:
        raise RuntimeError(f'Lossless bootstrap differs: {(baseline-candidate).abs().max().item()}')
    return {'mode':mode,'bootstrap_exact':exact,'codegen_modules':len(configs),
            'step_cache':True,'solver_constants':True,'noise_and_positions':True}
