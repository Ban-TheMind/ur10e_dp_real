"""Load upstream native checkpoints or standalone fine-tuning checkpoints."""

import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path, device="cpu", weights="ema", inference_steps=None):
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    # Upstream checkpoints are pickle files: load only trusted project/author weights.
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    if "policy_config" in payload:
        config = payload["policy_config"]
        state = payload["ema_model" if weights == "ema" else "model"]
    else:
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        config = OmegaConf.to_container(payload["cfg"].policy, resolve=True)
        state = payload["state_dicts"]["ema_model" if weights == "ema" else "model"]
    model = hydra.utils.instantiate(OmegaConf.create(config))
    model.load_state_dict(state, strict=True)
    if inference_steps is not None:
        if not 1 <= inference_steps <= model.noise_scheduler.config.num_train_timesteps:
            raise ValueError("Invalid number of inference denoising steps")
        model.num_inference_steps = inference_steps
    model.to(device).eval()
    return model, config, payload
