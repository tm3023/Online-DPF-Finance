"""E2 driver.

  python run_e2.py tune     # choose window L and learning rate on the tuning segment only
  python run_e2.py main     # pretrain on the tuning segment, walk forward on the evaluation segment
  python run_e2.py report   # baselines + table_main.md + fig_regime.png

Tuning protocol: within the tuning segment (first ~1 year), pretrain on its first 60% and
run online over its last 40%; pick (L, lr) with the highest mean one-step predictive
log-likelihood (NN residual on). Nothing from the evaluation segment is used.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

import dpf
from common import OUT, load_returns

SEEDS = [0, 1, 2, 3, 4]
GRID_L = [10, 20, 40]
GRID_LR = [1e-3, 3e-3, 1e-2]
PRETRAIN = dict(epochs=150, n=256, lr=0.02)
VARIANTS = {  # name: (use_nn, online)
    "dpf_online_nn": (True, True),
    "dpf_online_param": (False, True),
    "dpf_frozen_nn": (True, False),
    "pf_frozen_param": (False, False),   # = bootstrap PF on the parametric SV-t model (baseline)
}


def fresh_model(r_fit, use_nn, seed):
    torch.manual_seed(seed)
    return dpf.SVModel(mu0=2 * np.log(r_fit.std()), use_nn=use_nn)


def _pretrained(r_fit, use_nn, seed, tag):
    """Pretrain once per (segment, use_nn, seed) and checkpoint the fitted state to disk."""
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    f = os.path.join(OUT, "ckpt", f"pretrain_{tag}_nn{int(use_nn)}_seed{seed}.pt")
    m = fresh_model(r_fit, use_nn, seed)
    if os.path.exists(f):
        ck = torch.load(f)
        m.load_state_dict(ck["state"])
        return m, ck["hist"]
    t0 = time.time()
    hist = dpf.pretrain(m, r_fit, seed=seed, **PRETRAIN)
    torch.save({"state": m.state_dict(), "hist": hist}, f)
    print(f"  pretrain {tag} nn={use_nn} seed {seed}: {time.time()-t0:.0f}s, ll/obs {hist[-1]:.3f} (saved)", flush=True)
    return m, hist


def tune():
    r, dates, n_tune, _ = load_returns()
    cut = int(0.6 * n_tune)
    os.makedirs(OUT, exist_ok=True)
    rows_f = os.path.join(OUT, "tuning_rows.jsonl")  # one line per finished (seed, L, lr): resumable
    rows = [json.loads(x) for x in open(rows_f)] if os.path.exists(rows_f) else []
    done = {(x["seed"], x["L"], x["lr"]) for x in rows}
    for seed in SEEDS[:2]:
        base, _ = _pretrained(r[:cut], True, seed, "tune")
        state = {k: v.clone() for k, v in base.state_dict().items()}
        h0 = dpf.filter_through(base, r[:cut], 1000, torch.Generator().manual_seed(seed))
        for L, lr in itertools.product(GRID_L, GRID_LR):
            if (seed, L, lr) in done:
                continue
            m = fresh_model(r[:cut], True, seed)
            m.load_state_dict(state)
            o = dpf.walk_forward(m, r[:n_tune], cut, h0, online=True, L=L, lr=lr, seed=seed, crps=False)
            rows.append({"seed": seed, "L": L, "lr": lr, "mean_pred_ll": float(np.mean(o["pred_ll"]))})
            with open(rows_f, "a") as fh:
                fh.write(json.dumps(rows[-1]) + "\n")
            print(rows[-1], flush=True)
    agg = {}
    for row in rows:
        agg.setdefault((row["L"], row["lr"]), []).append(row["mean_pred_ll"])
    table = sorted(((k[0], k[1], float(np.mean(v))) for k, v in agg.items()), key=lambda x: -x[2])
    best = {"L": table[0][0], "lr": table[0][1], "mean_pred_ll": table[0][2]}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"protocol": f"pretrain on tuning[:{cut}], online on tuning[{cut}:{n_tune}]", "rows": rows,
               "ranked": [{"L": a, "lr": b, "mean_pred_ll": c} for a, b, c in table], "best": best},
              open(os.path.join(OUT, "tuning.json"), "w"), indent=2)
    print("BEST", best)


def main():
    r, dates, n_tune, _ = load_returns()
    best = json.load(open(os.path.join(OUT, "tuning.json")))["best"]
    L, lr = best["L"], best["lr"]
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    meta = {"L": L, "lr": lr, "pretrain": PRETRAIN, "n_tune": n_tune, "seeds": SEEDS,
            "n_params_nn=True": fresh_model(r[:n_tune], True, 0).n_params(),
            "n_params_nn=False": fresh_model(r[:n_tune], False, 0).n_params()}
    json.dump(meta, open(os.path.join(OUT, "main_meta.json"), "w"), indent=2)
    for seed in SEEDS:
        if all(os.path.exists(os.path.join(OUT, "runs", f"{n}_seed{seed}.npz")) for n in VARIANTS):
            continue  # this seed is fully done
        fitted = {}
        for use_nn in (True, False):
            m, hist = _pretrained(r[:n_tune], use_nn, seed, "main")   # checkpointed per seed
            fitted[use_nn] = ({k: v.clone() for k, v in m.state_dict().items()}, hist)
            meta[f"n_params_nn={use_nn}"] = m.n_params()
        for name, (use_nn, online) in VARIANTS.items():
            f = os.path.join(OUT, "runs", f"{name}_seed{seed}.npz")
            if os.path.exists(f):
                continue  # resume support: each finished (variant, seed) is saved immediately
            m = fresh_model(r[:n_tune], use_nn, seed)
            m.load_state_dict(fitted[use_nn][0])
            h0 = dpf.filter_through(m, r[:n_tune], 1000, torch.Generator().manual_seed(100 + seed))
            t0 = time.time()
            o = dpf.walk_forward(m, r, n_tune, h0, online=online, L=L, lr=lr, seed=seed)
            final = {"mu": m.mu.item(), "phi": m.phi.item(), "sig_h": m.sig_h.item(), "nu": m.nu.item()}
            np.savez(f, **o, pretrain_hist=np.asarray(fitted[use_nn][1]),
                     final_params=json.dumps(final))
            print(f"seed {seed} {name}: {time.time()-t0:.0f}s, mean pred_ll {np.mean(o['pred_ll']):.4f}, final {final}",
                  flush=True)
    json.dump(meta, open(os.path.join(OUT, "main_meta.json"), "w"), indent=2)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import baselines
    from common import EVENTS, block_bootstrap_ci

    r, dates, n_tune, mean_tune = load_returns()
    ev = slice(n_tune, len(r))
    d_ev = dates[ev]
    meta = json.load(open(os.path.join(OUT, "main_meta.json")))
    scores = {}
    # baselines (deterministic given the tuning-segment fit)
    ll, cr = baselines.ukf_scores(r, n_tune)
    scores["UKF (Tam, 5-state)"] = {"ll": ll[ev][None], "crps": cr[ev][None]}
    garch_params = {}
    for dist, lab in (("normal", "GARCH(1,1)-N"), ("t", "GARCH(1,1)-t")):
        ll, cr, p = baselines.garch_scores(r, n_tune, dist=dist)
        scores[lab] = {"ll": ll[ev][None], "crps": cr[ev][None]}
        garch_params[lab] = p
    ll, cr, plog = baselines.garch_t_rolling_scores(r, n_tune, refit_every=20)
    scores["GARCH(1,1)-t, refit every 20d"] = {"ll": ll[ev][None], "crps": cr[ev][None]}
    garch_params["GARCH(1,1)-t rolling (last refit)"] = {k: v for k, v in plog[-1].items() if k != "start"}
    labels = {"pf_frozen_param": "Bootstrap PF, SV-t (frozen)", "dpf_frozen_nn": "SV-t + NN residual (frozen)",
              "dpf_online_param": "SV-t, online", "dpf_online_nn": "SV-t + NN residual, online"}
    finals, inwin = {}, {}
    for name, lab in labels.items():
        runs = [np.load(os.path.join(OUT, "runs", f"{name}_seed{s}.npz")) for s in meta["seeds"]
                if os.path.exists(os.path.join(OUT, "runs", f"{name}_seed{s}.npz"))]
        if not runs:
            continue
        scores[lab] = {"ll": np.stack([x["pred_ll"] for x in runs]), "crps": np.stack([x["crps"] for x in runs])}
        finals[lab] = [json.loads(str(x["final_params"])) for x in runs]
        inwin[lab] = np.stack([x["inwin"] for x in runs])

    ref_pf = scores["Bootstrap PF, SV-t (frozen)"]["ll"].mean(0)
    ref_ukf = scores["UKF (Tam, 5-state)"]["ll"][0]
    rows = []
    for lab, sc in scores.items():
        per_seed = sc["ll"].mean(axis=1)
        mean_daily = sc["ll"].mean(0)
        d_pf = block_bootstrap_ci(mean_daily - ref_pf)
        d_ukf = block_bootstrap_ci(mean_daily - ref_ukf)
        d_rg = block_bootstrap_ci(mean_daily - scores["GARCH(1,1)-t, refit every 20d"]["ll"][0])
        rows.append({
            "model": lab, "seeds": int(sc["ll"].shape[0]),
            "pred_ll": float(per_seed.mean()), "pred_ll_sd_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
            "crps_x100": float(100 * sc["crps"].mean()),
            "d_vs_pf": d_pf, "d_vs_ukf": d_ukf, "d_vs_rolling_garch_t": d_rg,
        })
    rows.sort(key=lambda x: -x["pred_ll"])

    md = [f"# E2: BTC-USD one-step-ahead density forecasts (evaluation {d_ev[0].date()} – {d_ev[-1].date()}, {len(d_ev)} days)", "",
          f"Fit/tuning segment: {dates[0].date()} – {dates[n_tune-1].date()} ({n_tune} days); returns demeaned with its mean ({mean_tune:.5f}). "
          f"Online window L={meta['L']}, Adam lr={meta['lr']} (chosen on the tuning segment only). "
          f"Parameters: SV-t 4, SV-t+NN {meta['n_params_nn=True']}. Seeds: {meta['seeds']}. "
          "GARCH baselines: fitted once on the tuning year, or refitted every 20 days on all data to date (expanding window, no look-ahead).", "",
          "| Model | mean log score / day (± sd over seeds) | CRPS ×100 | Δ vs bootstrap PF [95% CI] | Δ vs UKF [95% CI] | Δ vs rolling GARCH-t [95% CI] |",
          "|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['model']} | {x['pred_ll']:.4f} ± {x['pred_ll_sd_seeds']:.4f} | {x['crps_x100']:.4f} | "
                  f"{x['d_vs_pf'][0]:+.4f} [{x['d_vs_pf'][1]:+.4f}, {x['d_vs_pf'][2]:+.4f}] | "
                  f"{x['d_vs_ukf'][0]:+.4f} [{x['d_vs_ukf'][1]:+.4f}, {x['d_vs_ukf'][2]:+.4f}] | "
                  f"{x['d_vs_rolling_garch_t'][0]:+.4f} [{x['d_vs_rolling_garch_t'][1]:+.4f}, {x['d_vs_rolling_garch_t'][2]:+.4f}] |")
    md += ["", "Higher log score and lower CRPS are better. CIs: moving-block bootstrap (block 20 days) on the daily difference "
           "(DPF variants averaged over seeds first).", ""]
    # overfitting probe: in-window (theta_t) minus next-day predictive (theta_t)
    gaps = {}
    for lab in ("SV-t + NN residual, online", "SV-t, online"):
        if lab in inwin:
            g = inwin[lab][:, :-1] - scores[lab]["ll"][:, 1:]
            gaps[lab] = g
            md.append(f"- Overfitting probe, {lab}: mean in-window − next-day log score = {np.nanmean(g):.4f} "
                      f"(seed sd {np.nanstd(np.nanmean(g, axis=1), ddof=1):.4f})")
    md.append("")
    md.append("Final parameters (seed 0): " + "; ".join(f"{k}: " + ", ".join(f"{a}={b:.3f}" for a, b in v[0].items())
                                                    for k, v in finals.items()))
    md.append("")
    md.append("GARCH parameters (tuning fit, % returns): " + "; ".join(f"{k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items())
                                                                  for k, v in garch_params.items()))
    open(os.path.join(OUT, "table_main.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # figure: returns, cumulative log-score differences, overfitting gap
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax[0].plot(d_ev, r[ev] * 100, lw=0.5, color="#555555")
    ax[0].set_ylabel("daily return (%)")
    ax[0].set_title("BTC-USD, evaluation segment")
    for lab, col in (("SV-t + NN residual, online", "#C0392B"), ("SV-t, online", "#2271B2"),
                     ("SV-t + NN residual (frozen)", "#D4853A"), ("UKF (Tam, 5-state)", "#2E8B57"),
                     ("GARCH(1,1)-t", "#888888"), ("GARCH(1,1)-t, refit every 20d", "#000000")):
        if lab in scores:
            ax[1].plot(d_ev, np.cumsum(scores[lab]["ll"].mean(0) - ref_pf), lw=1.2, color=col, label=lab)
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set_ylabel("cumulative Δ log score\nvs bootstrap PF (frozen)")
    ax[1].legend(fontsize=8, loc="best")
    for lab, col in (("SV-t + NN residual, online", "#C0392B"), ("SV-t, online", "#2271B2")):
        if lab in gaps:
            g = pd.Series(np.nanmean(gaps[lab], axis=0), index=d_ev[:-1]).rolling(30, min_periods=10).mean()
            ax[2].plot(g.index, g.values, color=col, lw=1.2, label=lab)
    ax[2].axhline(0, color="black", lw=0.6)
    ax[2].set_ylabel("in-window − next-day\nlog score (30d mean)")
    ax[2].legend(fontsize=8)
    for a in ax:
        for d, name in EVENTS.items():
            a.axvline(pd.Timestamp(d), color="grey", ls=":", lw=0.8)
    for d, name in EVENTS.items():
        ax[0].text(pd.Timestamp(d), ax[0].get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_regime.png"), dpi=150, bbox_inches="tight")
    json.dump({"rows": rows, "garch_params": garch_params, "final_params": finals},
              open(os.path.join(OUT, "table_main.json"), "w"), indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"tune": tune, "main": main, "report": report}[sys.argv[1]]()
