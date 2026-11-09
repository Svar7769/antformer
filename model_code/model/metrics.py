# Metrics on a whole split. Inputs are numpy; models are run through predict().
#
#   split_metrics(pred, lab)  real data (val/test): losses, accuracy per case, calibration, pick/drop
#   synth_metrics(pred, lab)  synth grid: accuracy per probe (all heads), calibration error
#
# Reference values ("ref/...") are the best any model can do on that split, because the
# explore coin makes trail vs wander random on smell rows.

import numpy as np
import torch

from dataset.config import CASES, PROBES, CASE_SMELL

LABEL_KEYS = ["mode", "direction", "pick", "drop", "case"]


@torch.no_grad()
def predict(net, dir_tok, self_tok, chunk=65536):
    """Run the model over all rows in chunks. Returns numpy probabilities."""
    was_training = net.training
    net.eval()
    outs = []
    for i in range(0, len(dir_tok), chunk):
        o = net(dir_tok[i:i + chunk], self_tok[i:i + chunk])
        outs.append({k: v.float().cpu() for k, v in o.items()})
    net.train(was_training)
    o = {k: torch.cat([x[k] for x in outs]) for k in outs[0]}
    return dict(
        mode_p=o["mode_logits"].softmax(-1).numpy(),         # (N, 3)
        dir_logp=o["dir_logits"].log_softmax(-1).numpy(),    # (N, 4)
        pick_p=torch.sigmoid(o["pick_logit"]).numpy(),       # (N,)
        drop_p=torch.sigmoid(o["drop_logit"]).numpy(),       # (N,)
    )


def _entropy(p):
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p))) if 0 < p < 1 else 0.0


def _bce(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _prec_rec(p, y):
    hat = p > 0.5
    tp = (hat & y).sum()
    prec = tp / hat.sum() if hat.sum() else float("nan")
    rec = tp / y.sum() if y.sum() else float("nan")
    return float(prec), float(rec)


def split_metrics(pred, lab, explore=0.25):
    m = {}
    mode, case, direction = lab["mode"], lab["case"], lab["direction"]
    mode_hat = pred["mode_p"].argmax(1)
    dir_hat = pred["dir_logp"].argmax(1)
    has_dir = direction >= 0
    n = len(mode)

    m["loss_mode"] = float(-np.log(np.clip(pred["mode_p"][np.arange(n), mode], 1e-12, 1)).mean())
    m["loss_dir"] = float(-pred["dir_logp"][has_dir, direction[has_dir]].mean()) if has_dir.any() else float("nan")
    m["loss_pick"] = _bce(pred["pick_p"], lab["pick"])
    m["loss_drop"] = _bce(pred["drop_p"], lab["drop"])
    m["loss"] = m["loss_mode"] + m["loss_dir"] + m["loss_pick"] + m["loss_drop"]

    m["mode_acc"] = float((mode_hat == mode).mean())
    m["dir_acc"] = float((dir_hat[has_dir] == direction[has_dir]).mean()) if has_dir.any() else float("nan")
    for i, c in enumerate(CASES):
        sel = case == i
        if not sel.any():
            continue
        m[f"mode_acc/{c}"] = float((mode_hat[sel] == mode[sel]).mean())
        m[f"p_wander/{c}"] = float(pred["mode_p"][sel, 2].mean())        # calibration
        d = sel & has_dir
        if d.any():
            m[f"dir_acc/{c}"] = float((dir_hat[d] == direction[d]).mean())

    m["pick_precision"], m["pick_recall"] = _prec_rec(pred["pick_p"], lab["pick"])
    m["drop_precision"], m["drop_recall"] = _prec_rec(pred["drop_p"], lab["drop"])

    frac_smell = float((case == CASE_SMELL).mean())
    m["ref/mode_acc_max"] = 1.0 - explore * frac_smell
    m["ref/loss_mode_min"] = frac_smell * _entropy(explore)
    m["ref/p_wander/smell"] = explore
    return m


def synth_metrics(pred, lab):
    """Accuracy per probe, counting a row correct only if every labelled head is right.
    Synth mode labels have the explore coin off, so on smell rows 'trail' is the right answer."""
    m = {}
    mode_hat = pred["mode_p"].argmax(1)
    dir_hat = pred["dir_logp"].argmax(1)
    has_dir = lab["direction"] >= 0
    ok = ((mode_hat == lab["mode"])
          & (~has_dir | (dir_hat == lab["direction"]))
          & ((pred["pick_p"] > 0.5) == lab["pick"])
          & ((pred["drop_p"] > 0.5) == lab["drop"]))
    for i, p in enumerate(PROBES):
        sel = lab["probe"] == i
        if sel.any():
            m[f"probe/{p}"] = float(ok[sel].mean())
    m["acc"] = float(ok.mean())
    m["calib_mae"] = float(np.abs(pred["mode_p"][:, 2] - lab["p_wander"]).mean())
    return m
