# Model settings. One place for every architecture choice.
# Input sizes come from dataset/config.py so the model always matches the data.

from dataclasses import dataclass, asdict, field

from dataset.config import DIR_FEATURES, SELF_FEATURES, OBS_DIM, DIRS, MODES


@dataclass
class ModelConfig:
    arch: str = "transformer"        # "transformer" or "mlp" (baseline)

    # ---- transformer ----
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 1
    use_layernorm: bool = False      # off: residual stream stays linear and easier to read
    dir_embedding: bool = True       # learned embedding per direction (up, down, left, right)
    dir_readout: str = "self"        # "self": direction logits from the self token
                                     # "pointer": one score per direction token
    act: str = "relu"                # MLP activation: "relu" or "gelu"

    # ---- MLP baseline ----
    mlp_hidden: tuple = (128, 128)

    # ---- sizes (from dataset/config.py, do not edit here) ----
    n_dir: int = len(DIRS)                     # 4 direction tokens
    n_dir_features: int = len(DIR_FEATURES)    # 5 features per direction token
    n_self_features: int = len(SELF_FEATURES)  # 3 features in the self token
    obs_dim: int = OBS_DIM                     # 23 flat inputs (MLP baseline)
    n_modes: int = len(MODES)                  # 3: target, trail, wander

    def __post_init__(self):
        assert self.arch in ("transformer", "mlp"), self.arch
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.dir_readout in ("self", "pointer"), self.dir_readout
        assert self.act in ("relu", "gelu"), self.act
        self.mlp_hidden = tuple(self.mlp_hidden)

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def n_tokens(self):
        return self.n_dir + 1            # 4 direction tokens + 1 self token

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


MODEL = ModelConfig()

# ---------------- training (python -m model.train) ----------------
from dataset.config import CFG as DATA_CFG


@dataclass
class TrainConfig:
    # data
    data_dir: str = DATA_CFG.out_dir
    train_fraction: float = 1.0      # < 1: train on a random subset (grokking experiments)
    balanced: bool = True            # sample by case (see BrainData.sample_weights)

    # optimisation
    steps: int = 20000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.1        # 1.0 underfit trail direction in short CPU tests; sweep 0, 0.1, 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 200          # linear warmup, then constant lr
    grad_clip: float = 1.0           # 0 = off

    # loss weights
    w_mode: float = 1.0
    w_dir: float = 1.0
    w_pick: float = 1.0
    w_drop: float = 1.0

    # evaluation, logging, checkpoints
    eval_every: int = 500
    eval_rows: int = 200_000         # val rows used at each eval (fixed random subset)
    n_ckpt: int = 30                 # log-spaced checkpoints (+ final)
    tensorboard: bool = False        # also write TensorBoard logs (needs `pip install tensorboard`)
    out_root: str = "outputs"
    run_name: str = ""               # empty: <arch>_wd<wd>_<timestamp>

    seed: int = 0
    device: str = "auto"             # "auto", "cuda", "cpu"

    def to_dict(self):
        return asdict(self)


TRAIN = TrainConfig()
