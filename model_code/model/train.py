# Train a brain model.
#
#   python -m model.train --run_name tf_wd1
#   python -m model.train --run_name mlp_base --arch mlp
#   python -m model.train --run_name tf_wd0 --weight_decay 0
#   python -m model.train --run_name tf_frac10 --train_fraction 0.1
#
# Every field of ModelConfig and TrainConfig (model/config.py) is a flag.
# Writes outputs/<run_name>/:
#   config.json      exact model + train config, dataset version
#   metrics.jsonl    one JSON line per evaluation (the training log)
#   ckpt/step_XXXXXXX.pt, ckpt/final.pt
#   tb/              TensorBoard logs (only with --tensorboard true)

import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from dataset.ant_dataset import BrainData
from model import build_model, count_params, save_model
from model.config import ModelConfig, TrainConfig
from model.losses import compute_loss
from model.metrics import predict, split_metrics, synth_metrics, LABEL_KEYS

SKIP_FLAGS = {"n_dir", "n_dir_features", "n_self_features", "obs_dim", "n_modes"}   # fixed by the data


# ---------------- config from command line ----------------
def _str2bool(s):
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {s}")


def add_flags(ap, cls):
    for f in dataclasses.fields(cls):
        if f.name in SKIP_FLAGS:
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if isinstance(default, bool):
            ap.add_argument(f"--{f.name}", type=_str2bool, default=None)
        elif isinstance(default, tuple):
            ap.add_argument(f"--{f.name}", type=int, nargs="+", default=None)
        else:
            ap.add_argument(f"--{f.name}", type=type(default), default=None)


def parse_configs(argv=None):
    ap = argparse.ArgumentParser(description="train a brain model")
    add_flags(ap, ModelConfig)
    add_flags(ap, TrainConfig)
    ap.add_argument("--overwrite", action="store_true", help="reuse an existing run folder")
    args = vars(ap.parse_args(argv))
    pick = lambda cls: {f.name: args[f.name] for f in dataclasses.fields(cls)
                        if f.name in args and args[f.name] is not None}
    return ModelConfig(**pick(ModelConfig)), TrainConfig(**pick(TrainConfig)), args["overwrite"]


# ---------------- data on device ----------------
def to_tensors(data, device):
    tok = data.tokens()
    t = dict(
        dir_tok=torch.as_tensor(tok["dir_tok"]),
        self_tok=torch.as_tensor(tok["self_tok"]),
        mode=torch.as_tensor(data.mode.astype(np.int64)),
        direction=torch.as_tensor(data.direction.astype(np.int64)),
        pick=torch.as_tensor(data.pick.astype(np.float32)),
        drop=torch.as_tensor(data.drop.astype(np.float32)),
    )
    return {k: v.to(device) for k, v in t.items()}


def labels(data):
    lab = {k: data.a[k] for k in LABEL_KEYS}
    for k in ("probe", "p_wander"):
        if k in data.a:
            lab[k] = data.a[k]
    return lab


def ckpt_schedule(steps, n):
    s = np.unique(np.geomspace(10, steps, n).astype(int)) if steps >= 10 else np.array([steps])
    return set(int(x) for x in s) | {steps}


def weight_norm(net):
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in net.parameters())))


# ---------------- evaluation ----------------
def evaluate(net, evalsets, explore):
    rec = {}
    for name, (T, lab) in evalsets.items():
        pred = predict(net, T["dir_tok"], T["self_tok"])
        m = synth_metrics(pred, lab) if name == "synth" else split_metrics(pred, lab, explore)
        rec.update({f"{name}/{k}": v for k, v in m.items()})
    return rec


def fmt_line(step, rec):
    g = lambda k, f="{:.3f}": f.format(rec[k]) if k in rec and not math.isnan(rec[k]) else "-"
    s = (f"step {step:7d} | train {g('train/loss')} | "
         f"val mode {g('val/loss_mode')} (min {g('val/ref/loss_mode_min')}) "
         f"dir_acc {g('val/dir_acc')} P(w|smell) {g('vl/p_wander/smell')}")
    if "synth/acc" in rec:
        s += (f" | synth {g('synth/acc')} behind {g('synth/probe/behind')} "
              f"tie {g('synth/probe/tie')} gate {g('synth/probe/gate_smell')}")
    return s


# ---------------- main ----------------
def main(argv=None):
    mcfg, tcfg, overwrite = parse_configs(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if tcfg.device == "auto" else tcfg.device
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    run_name = tcfg.run_name or f"{mcfg.arch}_wd{tcfg.weight_decay:g}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(tcfg.out_root, run_name)
    if os.path.exists(run_dir) and not overwrite:
        raise SystemExit(f"{run_dir} exists; pick another --run_name or pass --overwrite")
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

    # data
    tr = BrainData(tcfg.data_dir, "train")
    manifest = tr.manifest
    if tcfg.train_fraction < 1.0:
        rng = np.random.default_rng(tcfg.seed)
        keep = np.sort(rng.choice(len(tr), int(len(tr) * tcfg.train_fraction), replace=False))
        tr = tr.subset(keep)
    va = BrainData(tcfg.data_dir, "val")
    if len(va) > tcfg.eval_rows:
        va = va.subset(np.sort(np.random.default_rng(tcfg.seed + 1).choice(len(va), tcfg.eval_rows, replace=False)))

    T_tr = to_tensors(tr, device)
    evalsets = {"val": (to_tensors(va, device), labels(va))}
    if os.path.exists(os.path.join(tcfg.data_dir, "synth.npz")):
        sy = BrainData(tcfg.data_dir, "synth")
        evalsets["synth"] = (to_tensors(sy, device), labels(sy))
    weights = torch.as_tensor(tr.sample_weights(), dtype=torch.float32, device=device) if tcfg.balanced else None
    explore = manifest.get("sim_config", {}).get("explore", 0.25)

    # model, optimiser
    net = build_model(mcfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
                            betas=(tcfg.beta1, tcfg.beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, tcfg.warmup_steps)))

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(model=mcfg.to_dict(), train=tcfg.to_dict(), device=device,
                       n_params=count_params(net), train_rows=len(tr),
                       dataset_version=manifest.get("version"),
                       dataset_counts=manifest.get("counts")), f, indent=2)

    tb = None
    if tcfg.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except ImportError:
            print("tensorboard not installed, continuing without it (pip install tensorboard)")

    print(f"run {run_dir} | {mcfg.arch} {count_params(net):,} params | device {device} | "
          f"train rows {len(tr):,} | evals {list(evalsets)}")

    ckpts = ckpt_schedule(tcfg.steps, tcfg.n_ckpt)
    log = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    run_sum, run_n, t0 = None, 0, time.time()
    N = len(tr)

    for step in range(1, tcfg.steps + 1):
        if weights is not None:
            idx = torch.multinomial(weights, tcfg.batch_size, replacement=True)
        else:
            idx = torch.randint(0, N, (tcfg.batch_size,), device=device)
        b = {k: v[idx] for k, v in T_tr.items()}

        out = net(b["dir_tok"], b["self_tok"])
        loss, parts = compute_loss(out, b, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
        opt.step()
        sched.step()

        parts = {k: v.detach() for k, v in parts.items()}
        run_sum = parts if run_sum is None else {k: run_sum[k] + parts[k] for k in parts}
        run_n += 1

        if step == 1 or step % tcfg.eval_every == 0 or step == tcfg.steps:
            rec = dict(step=step, time=round(time.time() - t0, 1), lr=sched.get_last_lr()[0],
                       weight_norm=weight_norm(net))
            rec.update({f"train/{k}": float(v / run_n) for k, v in run_sum.items()})
            rec.update(evaluate(net, evalsets, explore))
            log.write(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                                  for k, v in rec.items()}) + "\n")
            log.flush()
            if tb is not None:
                for k, v in rec.items():
                    if k != "step" and isinstance(v, float) and not math.isnan(v):
                        tb.add_scalar(k, v, step)
            print(fmt_line(step, rec), flush=True)
            run_sum, run_n = None, 0

        if step in ckpts:
            save_model(net, os.path.join(run_dir, "ckpt", f"step_{step:07d}.pt"), step=step)

    save_model(net, os.path.join(run_dir, "ckpt", "final.pt"), step=tcfg.steps)
    log.close()
    if tb is not None:
        tb.close()
    print(f"done in {time.time() - t0:.0f}s, saved {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
