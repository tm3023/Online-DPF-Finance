"""
E1 summary: compare reproduced online-stage RMSEs with Table I of Li, Chen & Li (2023).

Reads logs/linear_dim_{dim}_lr_0.002_type_{trainType}/results.npz written by
test_losses.py (rmse_recorder: seeds x online windows) and reports two
aggregations, because the paper does not state which one its "mean +- std over
50 random simulations" uses:
  pooled   : mean / std over every (seed, window) RMSE
  per-seed : mean / std across seeds of each seed's average RMSE

Usage: python e1_summarise.py [--dim 2] [--out ../results/e1]
"""
import argparse
import json
import os

import numpy as np

TABLE_I = {  # Li, Chen & Li (2023), Table I: mean, std
    2: {"pretrain": (5.30, 3.51), "online": (1.83, 1.47), "supervised": (1.00, 1.23)},
    5: {"pretrain": (7.21, 2.86), "online": (3.14, 1.49), "supervised": (1.88, 1.21)},
    10: {"pretrain": (12.94, 3.94), "online": (6.12, 2.82), "supervised": (4.60, 2.10)},
}
LABEL = {"pretrain": "Pre-trained DPF", "online": "OL-DPF", "supervised": "DPF (oracle)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("..", "results", "e1"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, summary = [], {}
    for t in ["pretrain", "online", "supervised"]:
        f = os.path.join("logs", f"linear_dim_{a.dim}_lr_0.002_type_{t}", "results.npz")
        if not os.path.exists(f):
            continue
        rmse = np.asarray(np.load(f)["rmse_recorder"], dtype=float)
        per_seed = rmse.reshape(rmse.shape[0], -1).mean(axis=1)
        pm, ps = TABLE_I[a.dim][t]
        s = {
            "n_seeds": int(rmse.shape[0]), "n_windows": int(rmse.reshape(rmse.shape[0], -1).shape[1]),
            "pooled_mean": float(rmse.mean()), "pooled_std": float(rmse.std()),
            "per_seed_mean": float(per_seed.mean()), "per_seed_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
            "paper_mean": pm, "paper_std": ps,
            "pooled_mean_within_paper_1sd": bool(abs(rmse.mean() - pm) <= ps),
        }
        summary[t] = s
        rows.append(f"| {LABEL[t]} | {pm:.2f} ± {ps:.2f} | {s['pooled_mean']:.2f} ± {s['pooled_std']:.2f} | "
                    f"{s['per_seed_mean']:.2f} ± {(s['per_seed_std'] or float('nan')):.2f} | {s['n_seeds']} |")

    md = [f"# E1: Table I reproduction, d_X = {a.dim}", "",
          "| Method | Paper (Table I) | Reproduced, pooled (seed×window) | Reproduced, per-seed mean | seeds |",
          "|---|---|---|---|---|", *rows]
    with open(os.path.join(a.out, f"table1_dim{a.dim}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, f"table1_dim{a.dim}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    main()
