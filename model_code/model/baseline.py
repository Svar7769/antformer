# MLP baseline on the flat 23-dim obs. Same inputs and outputs as BrainTransformer,
# so training and metrics code can treat both the same way.

import torch
import torch.nn as nn

from model.config import ModelConfig


def tokens_to_flat(dir_tok, self_tok):
    """Inverse of BrainData.tokens(): back to the 23-dim obs layout."""
    B = dir_tok.shape[0]
    return torch.cat([self_tok, dir_tok.transpose(1, 2).reshape(B, -1)], dim=1)


class MLPBaseline(nn.Module):
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig(arch="mlp")
        layers, d_in = [], cfg.obs_dim
        for h in cfg.mlp_hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU() if cfg.act == "relu" else nn.GELU()]
            d_in = h
        self.body = nn.Sequential(*layers)
        self.head_mode = nn.Linear(d_in, cfg.n_modes)
        self.head_dir = nn.Linear(d_in, cfg.n_dir)
        self.head_pick = nn.Linear(d_in, 1)
        self.head_drop = nn.Linear(d_in, 1)

    def forward(self, dir_tok, self_tok, return_cache=False):
        x = tokens_to_flat(dir_tok, self_tok)
        h = self.body(x)
        out = dict(
            mode_logits=self.head_mode(h),
            dir_logits=self.head_dir(h),
            pick_logit=self.head_pick(h).squeeze(-1),
            drop_logit=self.head_drop(h).squeeze(-1),
        )
        return (out, {"input": x, "hidden": h}) if return_cache else out
