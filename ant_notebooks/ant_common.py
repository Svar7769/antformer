"""
ant_common.py: shared plumbing for the ant-epsilon notebooks.

Everything that more than one notebook needs lives here, so the notebooks only hold
the code that matters for their own result:

    settings and paths          (edit PROJECT / DATA_DIR here, or set env vars)
    state layout and the rule   (sees_target, signal, best_dir, n_ties, true_probs, ...)
    normalize                   (the ONE normalization used in training and in the simulator)
    data loading                (load_split, flatten)
    models                      (MLPBaseline, BrainModel, AntGPT, mixture_nll)
    sequences and windows       (build_sequences, all_windows, get_batch, train_seq_model, collect_seq)
    reading eps from a model    (implied_eps_*)
    hidden states and injection (run_with_hidden, run_with_injection)
    simulator access            (import_sim, run_sim)
    plotting                    (setup_plots, colors)

Notebook order: 01 -> 10. Each notebook saves what the next ones need into OUT_DIR.

Smoke-test mode: set the environment variable ANT_QUICK=1 to run every notebook end to
end in a few minutes on CPU with tiny settings (numbers are then meaningless).
"""
import inspect
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# Settings and paths
# =====================================================================================
PROJECT = Path(os.environ.get(
    "ANT_PROJECT", "/home/spatel/Desktop/Projects/collective_intelligence/ant_sim/simple_ant_sim"))
DATA_DIR = Path(os.environ.get("ANT_DATA_DIR", PROJECT / "data/dataset_20261025_104552_105991"))
OUT_DIR = Path(os.environ.get("ANT_OUT_DIR", PROJECT / "data/eps_models_v2"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("ANT_QUICK") == "1"          # smoke test only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS_GRID = np.round(np.arange(0.1, 1.01, 0.1), 1)   # the 10 eps values in the dataset

# per-step models (notebooks 02, 03)
STEP_BATCH, STEP_LR = 512, 1e-3
STEP_EPOCHS = 1 if QUICK else 2

# sequence models (notebooks 04 to 10)
SEQ_LEN, SEQ_STRIDE = 128, 64
SEQ_BATCH, SEQ_LR = 256, 3e-4
SEQ_EPOCHS = 1 if QUICK else 20
N_VAL_WINDOWS = 300 if QUICK else 3000              # windows used to score a sequence model
N_HIDDEN_WINDOWS = 300 if QUICK else 2000           # windows used for hidden-state analysis
N_PER_EPS = 40 if QUICK else 300                    # windows per eps for injection tests
LATE = slice(64, SEQ_LEN)                           # "long history" positions

# simulator
FOOD_PATCHES = [(80, 80), (20, 75)]                 # must match how the dataset runs placed food
SIM_STEPS = 60 if QUICK else 2000
SIM_SEEDS = range(2) if QUICK else range(10)


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def show(obj):
    """Print the source of a function or class (so notebooks can show shared code)."""
    print(inspect.getsource(obj))


# =====================================================================================
# State layout and the true rule
# =====================================================================================
# One state row = one ant at one step, 19 numbers:
#   [on_food, at_home, carrying, see_food(4), see_home(4), smell_food(4), smell_home(4)]
# Directions are always ordered up, down, left, right.
ON_FOOD, AT_HOME, CARRYING = 0, 1, 2
SEE_FOOD, SEE_HOME = slice(3, 7), slice(7, 11)
SMELL_FOOD, SMELL_HOME = slice(11, 15), slice(15, 19)
N_ACTIONS = 4


def sees_target(s):
    """True where the ant can see its current target (food if searching, home if carrying)."""
    carrying = s[:, CARRYING] > 0.5
    target = np.where(carrying[:, None], s[:, SEE_HOME], s[:, SEE_FOOD])
    return target.max(axis=1) > 0


def signal(s):
    """The 4 numbers the rule uses: the target if visible, otherwise the matching smell."""
    carrying = s[:, CARRYING] > 0.5
    target = np.where(carrying[:, None], s[:, SEE_HOME], s[:, SEE_FOOD])
    smell = np.where(carrying[:, None], s[:, SMELL_HOME], s[:, SMELL_FOOD])
    return np.where(sees_target(s)[:, None], target, smell)


def best_dir(s):
    """The rule's preferred direction (first one if tied)."""
    return signal(s).argmax(axis=1)


def n_ties(s):
    """How many directions share the top signal. 1 = clear choice, 4 = no signal at all."""
    sig = signal(s)
    return (sig == sig.max(axis=1, keepdims=True)).sum(axis=1)


def true_probs(s, eps):
    """True action odds: rule with prob 1 - eps (split over tied best dirs), uniform with prob eps."""
    sig = signal(s)
    is_best = sig == sig.max(axis=1, keepdims=True)
    k = is_best.sum(axis=1, keepdims=True)
    eps = np.broadcast_to(np.asarray(eps, dtype=np.float64), (len(s),)).reshape(-1, 1)
    return is_best * (1 - eps) / k + eps / N_ACTIONS


def true_entropy(s, eps):
    """Lowest possible loss per row (the 'floor'). Only reachable if the true eps is known."""
    p = true_probs(s, eps)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def normalize(s):
    """Works on any shape (..., 19). See values / 20, each smell block min-max scaled per row.
    Keeps the rule's answer unchanged (checked in notebook 01)."""
    s = s.astype(np.float32).copy()
    s[..., 3:11] /= 20.0
    for blk in (SMELL_FOOD, SMELL_HOME):
        block = s[..., blk]
        lo = block.min(axis=-1, keepdims=True)
        hi = block.max(axis=-1, keepdims=True)
        s[..., blk] = np.where(hi > lo, (block - lo) / np.maximum(hi - lo, 1e-8), 0.0)
    return s


# =====================================================================================
# Data loading
# =====================================================================================
def load_run(run_dir):
    """One run, kept in (T, N, ...) shape so each ant stays separate."""
    with np.load(run_dir / "traj.npz") as d:
        return dict(
            name=run_dir.name,
            state=d["state"].astype(np.float32),              # (T, N, 19) raw
            action=d["action"].astype(np.int64),              # (T, N)
            prev_action=d["prev_action"].astype(np.float32),  # (T, N, 4) one-hot, zeros at t=0
            pick=d["dec_pick"].astype(np.float32),            # (T, N)
            drop=d["dec_drop"].astype(np.float32),            # (T, N)
            delivered=d["delivered"],                         # (T+1,)
            eps=float(d["epsilon"]),
            seed=int(d["seed"]),
        )


def load_split(split, exclude_eps=()):
    """All runs of one split ('train', 'val' or 'test'), optionally leaving some eps out."""
    run_dirs = sorted(p.parent for p in (DATA_DIR / split).glob("*/traj.npz"))
    runs = [load_run(r) for r in run_dirs]
    return [r for r in runs if not np.isclose(r["eps"], exclude_eps).any()]


def flatten(run_list, norm=True):
    """One row per ant-step, for the per-step models."""
    state = np.concatenate([r["state"].reshape(-1, 19) for r in run_list])
    return dict(
        state=normalize(state) if norm else state,
        action=np.concatenate([r["action"].reshape(-1) for r in run_list]),
        pick=np.concatenate([r["pick"].reshape(-1) for r in run_list]),
        drop=np.concatenate([r["drop"].reshape(-1) for r in run_list]),
        eps=np.concatenate([np.full(r["action"].size, r["eps"], dtype=np.float32) for r in run_list]),
    )


# =====================================================================================
# Models
# =====================================================================================
class MLPBaseline(nn.Module):
    """V1: plain MLP on one state. No eps, no history."""
    def __init__(self, d_in=19, hidden=128):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.action_head = nn.Linear(hidden, N_ACTIONS)
        self.pick_head = nn.Linear(hidden, 1)
        self.drop_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.ffn(x)
        return self.action_head(z), self.pick_head(z).squeeze(-1), self.drop_head(z).squeeze(-1)


class BrainModel(nn.Module):
    """V2: the 'brain'. Scores each direction with ONE shared small network, using only the
    blocks the rule uses (target and smell, picked by 'carrying'). With no signal all four
    directions look identical, so the output is exactly uniform."""
    def __init__(self, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))
        self.pick_head = nn.Linear(3, 1)
        self.drop_head = nn.Linear(3, 1)

    def forward(self, x):                                                    # x: (B, 19) normalized
        flags = x[:, :3]
        carrying = flags[:, CARRYING:CARRYING + 1] > 0.5                     # (B, 1)
        target = torch.where(carrying, x[:, SEE_HOME], x[:, SEE_FOOD])       # (B, 4)
        smell = torch.where(carrying, x[:, SMELL_HOME], x[:, SMELL_FOOD])    # (B, 4)
        per_dir = torch.stack([target, smell], dim=-1)                       # (B, 4, 2)
        tokens = torch.cat([per_dir, flags[:, None, :].expand(-1, 4, -1)], dim=-1)  # (B, 4, 5)
        a_logit = self.score(tokens).squeeze(-1)                             # (B, 4)
        return a_logit, self.pick_head(flags).squeeze(-1), self.drop_head(flags).squeeze(-1)


def mixture_nll(a_logit, action, eps):
    """-log P(action) when the ant follows the brain with prob 1 - eps and moves uniformly
    with prob eps. eps is the TRUE eps of the run, used in training only."""
    p_brain = a_logit.softmax(-1).gather(1, action[:, None]).squeeze(1)
    return -torch.log((1 - eps) * p_brain + eps / N_ACTIONS + 1e-12)


class AntGPT(nn.Module):
    """Small causal transformer: step t only sees steps 0..t. Never given eps."""
    def __init__(self, d_in, d_model=64, n_layers=3, n_heads=4, max_len=SEQ_LEN):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, N_ACTIONS)

    def forward(self, x):                                    # x: (B, T, d_in)
        T = x.shape[1]
        h = self.inp(x) + self.pos(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.out(self.norm(h))                        # (B, T, 4) logits


def load_brain():
    """The trained brain from notebook 03."""
    model = BrainModel().to(DEVICE)
    model.load_state_dict(torch.load(OUT_DIR / "brain_model.pt", map_location=DEVICE))
    return model.eval()


def load_seq_model(name):
    """A trained sequence model saved by notebooks 04 to 06 (or 09) as seq_<name>.pt."""
    ckpt = torch.load(OUT_DIR / f"seq_{name}.pt", map_location=DEVICE)
    model = AntGPT(d_in=ckpt["d_in"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def save_seq_model(model, name, d_in):
    torch.save(dict(state_dict=model.state_dict(), d_in=d_in), OUT_DIR / f"seq_{name}.pt")


# =====================================================================================
# Sequences, windows, training
# =====================================================================================
@torch.no_grad()
def brain_probs(brain, states_norm, batch=65536):
    """The brain's odds for each state. Input (..., 19) normalized, output (..., 4)."""
    brain.eval()
    shape = states_norm.shape[:-1]
    flat_s = torch.from_numpy(np.ascontiguousarray(states_norm.reshape(-1, 19)))
    out = []
    for start in range(0, len(flat_s), batch):
        a_logit, _, _ = brain(flat_s[start:start + batch].to(DEVICE))
        out.append(a_logit.softmax(-1).cpu().numpy())
    return np.concatenate(out).reshape(*shape, 4).astype(np.float32)


def build_sequences(run_list, with_actions=True, brain=None, shuffle=False, seed=0):
    """One long sequence per ant. Returns X (ants, T, d), Y (ants, T), EPS (ants,).
    Columns of X: [normalized state (19) | brain odds (4) if brain | previous action (4) if with_actions].
    The state is always the first 19 columns. EPS is kept for analysis only, never fed to a model.
    shuffle=True reorders each ant's steps at random (keeps each state paired with its action)."""
    assert not (shuffle and with_actions), "previous action means nothing once time is shuffled"
    rng = np.random.default_rng(seed)
    X, Y, EPS = [], [], []
    for r in run_list:
        s = normalize(r["state"])                                   # (T, N, 19)
        parts = [s]
        if brain is not None:
            parts.append(brain_probs(brain, s))
        if with_actions:
            parts.append(r["prev_action"])
        x = np.concatenate(parts, axis=-1).transpose(1, 0, 2)       # (N, T, d)
        y = r["action"].T                                           # (N, T)
        if shuffle:
            order = np.argsort(rng.random(y.shape), axis=1)         # a different order per ant
            x = np.take_along_axis(x, order[..., None], axis=1)
            y = np.take_along_axis(y, order, axis=1)
        X.append(np.ascontiguousarray(x, dtype=np.float32))
        Y.append(np.ascontiguousarray(y))
        EPS.append(np.full(x.shape[0], r["eps"], dtype=np.float32))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(EPS)


def all_windows(n_ants, T, seq_len=SEQ_LEN, stride=SEQ_STRIDE):
    """Every (ant, start) pair. One window = seq_len consecutive steps of one ant."""
    starts = np.arange(0, T - seq_len + 1, stride)
    ants, t0 = np.meshgrid(np.arange(n_ants), starts, indexing="ij")
    return ants.ravel(), t0.ravel()


def get_batch(X, Y, EPS, ants, t0, seq_len=SEQ_LEN):
    t = t0[:, None] + np.arange(seq_len)[None, :]
    return torch.from_numpy(X[ants[:, None], t]), torch.from_numpy(Y[ants[:, None], t]), EPS[ants]


def train_seq_model(model, data, epochs=SEQ_EPOCHS, batch=SEQ_BATCH, lr=SEQ_LR):
    """Plain next-action cross-entropy. eps is never used."""
    X, Y, EPS = data
    ants, t0 = all_windows(len(X), X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(ants))
        start_time, total, count = time.time(), 0.0, 0
        for start in range(0, len(order), batch):
            sel = order[start:start + batch]
            x, y, _ = get_batch(X, Y, EPS, ants[sel], t0[sel])
            loss = F.cross_entropy(model(x.to(DEVICE)).reshape(-1, N_ACTIONS), y.reshape(-1).to(DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            count += 1
        losses.append(total / count)
        print(f"   epoch {epoch:>2}: loss {losses[-1]:.4f}   ({time.time() - start_time:.0f} s)")
    model.eval()
    return losses


@torch.no_grad()
def collect_seq(model, data, n_windows=N_VAL_WINDOWS, batch=256, seed=0):
    """Run a model on fixed random windows. Returns probs (W, L, 4), actions (W, L),
    normalized states (W, L, 19), eps (W,). Same seed = same windows for every model."""
    X, Y, EPS = data
    ants, t0 = all_windows(len(X), X.shape[1])
    keep = np.random.default_rng(seed).choice(len(ants), size=min(n_windows, len(ants)), replace=False)
    ants, t0 = ants[keep], t0[keep]
    model.eval()
    probs, acts, states, eps = [], [], [], []
    for start in range(0, len(ants), batch):
        x, y, e = get_batch(X, Y, EPS, ants[start:start + batch], t0[start:start + batch])
        probs.append(model(x.to(DEVICE)).softmax(-1).cpu().numpy())
        acts.append(y.numpy())
        states.append(x[..., :19].numpy())
        eps.append(e)
    return np.concatenate(probs), np.concatenate(acts), np.concatenate(states), np.concatenate(eps)


# =====================================================================================
# Reading eps from a model's output
# =====================================================================================
# On a clear-choice step the true chance of the best move is 1 - 0.75 * eps,
# so the eps a model "believes" is (1 - p_best) / 0.75.
def implied_eps_steps(probs, states):
    """Per step: implied eps (W, L) and a mask of clear-choice steps (W, L)."""
    W, L, _ = probs.shape
    s = states.reshape(-1, 19)
    clear = (n_ties(s) == 1).reshape(W, L)
    best = best_dir(s).reshape(W, L)
    p_best = np.take_along_axis(probs, best[..., None], axis=-1)[..., 0]
    return np.clip((1 - p_best) / 0.75, 0, 1), clear


def implied_eps_windows(probs, states, sl=LATE):
    """Per window: implied eps averaged over its clear-choice steps in positions sl."""
    e, clear = implied_eps_steps(probs, states)
    e, clear = e[:, sl], clear[:, sl]
    n = clear.sum(1)
    return np.where(n > 0, (e * clear).sum(1) / np.maximum(n, 1), np.nan)


def rule_agreement(probs, states, sl=LATE):
    """Share of clear-choice steps (positions sl) where the model's top move is the rule's move."""
    W, L, _ = probs.shape
    s = states.reshape(-1, 19)
    clear = (n_ties(s) == 1).reshape(W, L)[:, sl]
    best = best_dir(s).reshape(W, L)[:, sl]
    return safe_mean((probs.argmax(-1)[:, sl] == best)[clear])


def safe_mean(x):
    """Mean that returns NaN (without a warning) when there is nothing to average."""
    x = np.asarray(x)
    return x.mean() if x.size else np.nan


def safe_nanmean(x):
    """nanmean that returns NaN (without a warning) when every value is NaN."""
    x = np.asarray(x, dtype=float)
    return np.nanmean(x) if np.isfinite(x).any() else np.nan


BUCKETS = {"0": slice(0, 1), "1-7": slice(1, 8), "8-31": slice(8, 32),
           "32-63": slice(32, 64), "64-127": slice(64, SEQ_LEN)}
RULE_EPS_MAX = 0.7   # near eps 1.0 the right prediction is almost uniform, so "agrees with rule" means little


def summarize_seq_model(probs, acts, states, eps):
    """Scores one sequence model on its validation windows.
    Returns by_history (gap and rule agreement per history range),
            by_eps (implied eps per true eps per history range), scores (headline numbers)."""
    W, L, _ = probs.shape
    s = states.reshape(-1, 19)
    floor = true_entropy(s, np.repeat(eps, L)).reshape(W, L)
    nll = -np.log(np.take_along_axis(probs, acts[..., None], axis=-1)[..., 0] + 1e-12)
    gap = nll - floor                                              # loss above the best possible

    eps_hat, clear = implied_eps_steps(probs, states)
    best = best_dir(s).reshape(W, L)
    rule_ok = probs.argmax(-1) == best
    rule_rows = clear & (eps <= RULE_EPS_MAX + 1e-6)[:, None]

    by_history = pd.DataFrame([
        dict(history=name, gap=gap[:, sl].mean(), rule_agreement=safe_mean(rule_ok[:, sl][rule_rows[:, sl]]))
        for name, sl in BUCKETS.items()]).set_index("history")

    rows = []
    for e in np.unique(eps):
        m = np.isclose(eps, e)
        row = {"true_eps": round(float(e), 1)}
        for name, sl in BUCKETS.items():
            row[name] = safe_mean(eps_hat[m, sl][clear[m, sl]])
        rows.append(row)
    by_eps = pd.DataFrame(rows).set_index("true_eps")

    late = by_eps["64-127"]
    true = late.index.values
    clean = true <= RULE_EPS_MAX + 1e-6
    ok = np.isfinite(late.values)
    slope, intercept = np.polyfit(true[ok], late.values[ok], 1) if ok.sum() >= 2 else (np.nan, np.nan)

    # best possible gap with NO history: true odds averaged over all eps values
    p_mix = np.mean([true_probs(s, e) for e in np.unique(eps)], axis=0)
    nll_mix = -np.log(p_mix[np.arange(len(s)), acts.reshape(-1)] + 1e-12)

    scores = {
        "gap, no-history ideal": (nll_mix - floor.reshape(-1)).mean(),
        "gap, history 0": gap[:, 0].mean(),
        "gap, history 64-127": gap[:, LATE].mean(),
        "rule agreement, 64-127 (eps<=0.7)": by_history.loc["64-127", "rule_agreement"],
        "eps error, all eps": np.abs(late.values - true).mean(),
        "eps error, eps<=0.7": np.abs(late.values[clean] - true[clean]).mean(),
        "eps slope (want 1)": slope,
        "eps intercept (want 0)": intercept,
    }
    return by_history, by_eps, scores


# =====================================================================================
# Hidden states and injection (notebooks 07 to 10)
# =====================================================================================
STAGES = ["input embedding", "after layer 1", "after layer 2", "after layer 3"]


@torch.no_grad()
def run_with_injection(model, x, stage=None, vec=None, keep_hidden=False):
    """Same computation as model(x), done stage by stage.
    stage 0 = input embedding, 1-3 = after that layer. vec is added to the hidden state at
    every step: shape (64,) for all windows, or (B, 64) for one vector per window.
    Returns action probs (B, T, 4), plus the hidden state after every stage if keep_hidden."""
    T = x.shape[1]
    h = model.inp(x) + model.pos(torch.arange(T, device=x.device))
    mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
    v = None
    if vec is not None:
        v = torch.as_tensor(np.asarray(vec), dtype=h.dtype, device=h.device)
        if v.dim() == 2:
            v = v[:, None, :]
    if stage == 0 and v is not None:
        h = h + v
    hidden = [h]
    for i, layer in enumerate(model.blocks.layers, start=1):
        h = layer(h, src_mask=mask, is_causal=True)
        if stage == i and v is not None:
            h = h + v
        hidden.append(h)
    probs = model.out(model.norm(h)).softmax(-1)
    return (probs, hidden) if keep_hidden else probs.cpu().numpy()


@torch.no_grad()
def collect_hidden(model, data, n_windows=N_HIDDEN_WINDOWS, seed=0, batch=256):
    """Hidden state after every stage for fixed random windows.
    Returns hidden (stages, W, L, 64) float16, probs (W, L, 4), states (W, L, 19), eps (W,)."""
    X, Y, EPS = data
    ants, t0 = all_windows(len(X), X.shape[1])
    keep = np.random.default_rng(seed).choice(len(ants), size=min(n_windows, len(ants)), replace=False)
    ants, t0 = ants[keep], t0[keep]
    H, P, S, E = [], [], [], []
    for start in range(0, len(ants), batch):
        x, _, e = get_batch(X, Y, EPS, ants[start:start + batch], t0[start:start + batch])
        probs, hidden = run_with_injection(model, x.to(DEVICE), keep_hidden=True)
        H.append(torch.stack(hidden).cpu().numpy().astype(np.float16))
        P.append(probs.cpu().numpy())
        S.append(x[..., :19].numpy())
        E.append(e)
    return dict(hidden=np.concatenate(H, axis=1), probs=np.concatenate(P),
                states=np.concatenate(S), eps=np.concatenate(E))


def windows_for_eps(data, eps_value, n=N_PER_EPS, seed=0):
    """n random windows from ants whose true eps is eps_value. Returns model input and states."""
    X, Y, EPS = data
    ants, t0 = all_windows(len(X), X.shape[1])
    pool = np.flatnonzero(np.isclose(EPS[ants], eps_value))
    pick = np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False)
    x, _, _ = get_batch(X, Y, EPS, ants[pick], t0[pick])
    return x.to(DEVICE), x[..., :19].numpy()


def random_like(vec, rng):
    """Control: a random direction with the same length as vec."""
    if np.linalg.norm(vec) == 0:
        return vec.copy()
    r = rng.normal(size=vec.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(vec)


def eps_means(hidden_stage, eps):
    """Average hidden vector per eps over long-history positions. hidden_stage: (W, L, 64).
    Returns dict eps -> (64,) vector. These are the 'eps vectors' used for injection."""
    Z = hidden_stage[:, LATE].astype(np.float32).mean(axis=1)
    return {round(float(e), 1): Z[np.isclose(eps, e)].mean(0) for e in np.unique(eps)}


def natural_reading(model, data, stage=2, seed=0):
    """Implied eps the model reaches on its own (no injection), for every eps."""
    out = {}
    for e in EPS_GRID:
        x_e, s_e = windows_for_eps(data, e, seed=seed)
        out[round(float(e), 1)] = safe_nanmean(implied_eps_windows(run_with_injection(model, x_e), s_e))
    return out


def knob_error(model, means, data, goals, alpha, source, targets, stage=2, seed=0):
    """Average |implied - goal| when injecting alpha * (means[target] - means[source])
    into windows whose true eps is source."""
    x_s, s_s = windows_for_eps(data, source, seed=seed)
    errs = []
    for t in targets:
        p = run_with_injection(model, x_s, stage, alpha * (means[t] - means[source]))
        errs.append(abs(safe_nanmean(implied_eps_windows(p, s_s)) - goals[t]))
    return float(np.mean(errs))


def ridge_fit(Z, y, lam=10.0):
    """Linear probe with standardized inputs. Returns a function Z_new -> y_hat."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + lam * np.eye(Z.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Znew: ((Znew - mu) / sd) @ w + y.mean()


def r2(y, y_hat):
    return 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# =====================================================================================
# Simulator access (notebooks 03 and 10)
# =====================================================================================
def import_sim():
    """Returns (sim_mod, Simulation, Config). sim_mod is the module where step_ant looks up
    `decide`, so replacing sim_mod.decide swaps the ant brain."""
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from simulation import Simulation
    from config import Config
    return sys.modules[Simulation.__module__], Simulation, Config


def run_sim(eps, seed, decide_fn, T=SIM_STEPS):
    """Food delivered after T steps with the given brain. Always restores the real brain."""
    sim_mod, Simulation, Config = import_sim()
    original = sim_mod.decide
    sim_mod.decide = decide_fn
    try:
        sim = Simulation(Config(epsilon=eps, seed=seed))
        for x, y in FOOD_PATCHES:
            sim.world.add_food(x, y, radius=4, amount_per_cell=5)
        for _ in range(T):
            sim.step()
        return sim.world.food_delivered
    finally:
        sim_mod.decide = original


# =====================================================================================
# Plotting
# =====================================================================================
# Fixed categorical order (never cycled), one ramp for ordered things like history length.
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", gray="#8a8a85", ink="#2b2b28")
CAT = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"]]
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]   # light = short, dark = long


def setup_plots():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ad", "axes.labelcolor": C["ink"], "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "xtick.color": "#5c5b56", "ytick.color": "#5c5b56",
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": plt.cycler(color=CAT),
    })


def diagonal(ax, label="perfect"):
    """y = x reference line for 'implied vs true' plots."""
    ax.plot([0, 1], [0, 1], ls="--", color=C["gray"], lw=1, label=label, zorder=0)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
