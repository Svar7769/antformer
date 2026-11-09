# model package: config, transformer, MLP baseline, and a factory.
#
#   from model import build_model, ModelConfig
#   net = build_model(ModelConfig(d_model=64))
#
# Self-test:  python -m model            (add --overfit for a quick learnability check)

import torch

from model.config import ModelConfig, MODEL
from model.transformer import BrainTransformer
from model.baseline import MLPBaseline, tokens_to_flat


def build_model(cfg: ModelConfig = None):
    cfg = cfg or MODEL
    return BrainTransformer(cfg) if cfg.arch == "transformer" else MLPBaseline(cfg)


def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def save_model(net, path, **extra):
    """Weights + config in one file, so a checkpoint rebuilds itself."""
    torch.save(dict(model_cfg=net.cfg.to_dict(), state_dict=net.state_dict(), **extra), path)


def load_model(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    net = build_model(ModelConfig.from_dict(ck["model_cfg"]))
    net.load_state_dict(ck["state_dict"])
    return net, ck
