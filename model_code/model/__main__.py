# Self-test for the model package. Does not need the training code.
#
#   python -m model                       shapes, attention, gradients, real-data round trip
#   python -m model --overfit             + quick check that each model can learn the synth grid
#   python -m model --data data/brain_v0  (default: dataset CFG.out_dir)

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset.config import CFG, PROBES
from model import ModelConfig, build_model, count_params, tokens_to_flat, save_model, load_model


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


def random_batch(cfg, B=8):
    return torch.rand(B, cfg.n_dir, cfg.n_dir_features), torch.randint(0, 2, (B, cfg.n_self_features)).float()


def test_shapes(net, cfg, B=8):
    dir_tok, self_tok = random_batch(cfg, B)
    out, cache = net(dir_tok, self_tok, return_cache=True)
    check(out["mode_logits"].shape == (B, cfg.n_modes), f"mode_logits {tuple(out['mode_logits'].shape)}")
    check(out["dir_logits"].shape == (B, cfg.n_dir), f"dir_logits {tuple(out['dir_logits'].shape)}")
    check(out["pick_logit"].shape == (B,) and out["drop_logit"].shape == (B,), "pick/drop logits (B,)")
    check(all(torch.isfinite(v).all() for v in out.values()), "outputs finite")
    return out, cache


def test_transformer_cache(cache, cfg, B=8):
    T = cfg.n_tokens
    check(cache["embed"].shape == (B, T, cfg.d_model), f"embed {tuple(cache['embed'].shape)}")
    for i in range(cfg.n_layers):
        p = cache[f"blocks.{i}.attn_pattern"]
        check(p.shape == (B, cfg.n_heads, T, T), f"layer {i} attn_pattern {tuple(p.shape)}")
        check(torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5), f"layer {i} attention rows sum to 1")
        check(cache[f"blocks.{i}.mlp_post"].shape == (B, T, cfg.d_mlp), f"layer {i} mlp_post {tuple(cache[f'blocks.{i}.mlp_post'].shape)}")
    print("  cache keys:", ", ".join(cache))


def test_grads(net, cfg):
    dir_tok, self_tok = random_batch(cfg)
    out = net(dir_tok, self_tok)
    sum(v.sum() for v in out.values()).backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    check(not missing, "every parameter gets a gradient" + (f" (missing: {missing})" if missing else ""))
    net.zero_grad()


def test_save_load(net, cfg, tmp="/tmp/_model_selftest.pt"):
    dir_tok, self_tok = random_batch(cfg)
    save_model(net, tmp)
    net2, _ = load_model(tmp)
    a, b = net(dir_tok, self_tok), net2(dir_tok, self_tok)
    check(all(torch.allclose(a[k], b[k]) for k in a), "save_model / load_model round trip")
    os.remove(tmp)


def test_real_data(nets, data_dir):
    from dataset.ant_dataset import BrainData
    split = "synth" if os.path.exists(os.path.join(data_dir, "synth.npz")) else "val"
    d = BrainData(data_dir, split)
    idx = np.random.default_rng(0).choice(len(d), 256, replace=False)
    sub = d.subset(idx)
    tok = sub.tokens()
    dir_tok, self_tok = torch.as_tensor(tok["dir_tok"]), torch.as_tensor(tok["self_tok"])
    flat = tokens_to_flat(dir_tok, self_tok).numpy()
    check(np.array_equal(flat, sub.obs), f"tokens -> flat obs round trip on real {split} rows")
    for name, net in nets.items():
        out = net(dir_tok, self_tok)
        check(all(torch.isfinite(v).all() for v in out.values()), f"{name} forward on real {split} rows")
    return d


def overfit(net, d, steps=600, n=4000, lr=3e-3, seed=0):
    """Train briefly on synth rows and report per-probe accuracy. Not real training:
    only checks the model has the capacity and wiring to learn every rule."""
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).choice(len(d), min(n, len(d)), replace=False)
    sub = d.subset(idx)
    tok = sub.tokens()
    X = torch.as_tensor(tok["dir_tok"]), torch.as_tensor(tok["self_tok"])
    y_mode = torch.as_tensor(sub.mode.astype(np.int64))
    y_dir = torch.as_tensor(sub.direction.astype(np.int64))
    y_pick = torch.as_tensor(sub.pick.astype(np.float32))
    y_drop = torch.as_tensor(sub.drop.astype(np.float32))
    has_dir = y_dir >= 0

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    t0 = time.time()
    for step in range(steps):
        out = net(*X)
        loss = (F.cross_entropy(out["mode_logits"], y_mode)
                + F.cross_entropy(out["dir_logits"][has_dir], y_dir[has_dir])
                + F.binary_cross_entropy_with_logits(out["pick_logit"], y_pick)
                + F.binary_cross_entropy_with_logits(out["drop_logit"], y_drop))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = net(*X)
    pm = out["mode_logits"].argmax(-1)
    pd = out["dir_logits"].argmax(-1)
    ok = (pm == y_mode) & (~has_dir | (pd == y_dir)) \
        & ((out["pick_logit"] > 0) == y_pick.bool()) & ((out["drop_logit"] > 0) == y_drop.bool())
    probe = torch.as_tensor(sub.probe.astype(np.int64))
    print(f"  loss {loss.item():.4f} after {steps} steps ({time.time() - t0:.1f}s), "
          f"all-heads accuracy on the rows it trained on:")
    print("  " + "  ".join(f"{p} {ok[probe == i].float().mean().item():.2f}" for i, p in enumerate(PROBES)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=CFG.out_dir)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    torch.manual_seed(0)

    cfgs = {"transformer": ModelConfig(arch="transformer"), "mlp": ModelConfig(arch="mlp")}
    nets = {k: build_model(c) for k, c in cfgs.items()}

    for name, net in nets.items():
        cfg = cfgs[name]
        print(f"\n[{name}]  params {count_params(net):,}")
        _, cache = test_shapes(net, cfg)
        if name == "transformer":
            test_transformer_cache(cache, cfg)
        test_grads(net, cfg)
        test_save_load(net, cfg)

    d = None
    if os.path.isdir(args.data):
        print(f"\n[real data: {args.data}]")
        d = test_real_data(nets, args.data)
    else:
        print(f"\n[real data] skipped, {args.data} not found")

    if args.overfit:
        if d is None or "probe" not in d.a:
            print("\n[overfit] skipped, needs synth.npz (python -m dataset.synth)")
        else:
            for name in nets:
                torch.manual_seed(0)
                net = build_model(cfgs[name])
                print(f"\n[overfit {name}]")
                overfit(net, d, steps=args.steps)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
