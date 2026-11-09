# Score a trained run on any split.
#
#   python -m model.evaluate outputs/tf_wd1                       val + synth, final checkpoint
#   python -m model.evaluate outputs/tf_wd1 --ckpt step_0001000.pt
#   python -m model.evaluate outputs/tf_wd1 --splits test         test: only once, at the very end
#
# Prints a table and writes eval_<ckpt>_<split>.json into the run folder.

import argparse
import json
import os

import torch

from dataset.ant_dataset import BrainData
from model import load_model
from model.metrics import predict, split_metrics, synth_metrics
from model.train import to_tensors, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--splits", nargs="+", default=["val", "synth"])
    ap.add_argument("--data_dir", default=None, help="default: data_dir saved in the run's config.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    data_dir = args.data_dir or cfg["train"]["data_dir"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    net, ck = load_model(os.path.join(args.run_dir, "ckpt", args.ckpt), map_location=device)
    net.to(device)
    print(f"{args.run_dir}  ckpt {args.ckpt} (step {ck.get('step')})  data {data_dir}")

    for split in args.splits:
        d = BrainData(data_dir, split)
        T = to_tensors(d, device)
        pred = predict(net, T["dir_tok"], T["self_tok"])
        if split == "synth":
            m = synth_metrics(pred, labels(d))
        else:
            explore = d.manifest.get("sim_config", {}).get("explore", 0.25)
            m = split_metrics(pred, labels(d), explore)
        print(f"\n[{split}]  {len(d):,} rows")
        for k, v in m.items():
            print(f"  {k:28s} {v:.4f}")
        out = os.path.join(args.run_dir, f"eval_{os.path.splitext(args.ckpt)[0]}_{split}.json")
        json.dump(m, open(out, "w"), indent=2)
    print()


if __name__ == "__main__":
    main()
