# E2: online differentiable particle filtering on BTC-USD

Extends the OL-DPF idea (Li, Chen & Li 2023; reproduced in E1) to a real, heavy-tailed financial series, and asks whether online, unsupervised learning of a particle filter helps daily volatility density forecasts and whether a neural component overfits.

## Setup
- **Data:** BTC-USD daily close, 2021-10-02 → 2026-10-02 (same file as the Bitcoin-UKF E0 diagnostics; `data/` is gitignored, sha256 `c60148f4…`), 1,826 log-returns.
- **Split:** first 364 days (to 2022-10-01) for all fitting and tuning; walk-forward evaluation on the remaining 1,462 days (2022-10-02 → 2026-10-02), which include FTX, SVB/USDC, the August 2024 sell-off and the February 2026 crash.
- **Model:** log-volatility AR(1) with standardised Student-t returns (SV-t, 4 parameters), optionally plus a zero-initialised 1-8-1 tanh residual on the transition (29 parameters). Bootstrap particle filter with reparameterised transitions; multinomial resampling without differentiating the indices.
- **Online learning:** after each day, one Adam step on the sliding-window (L days) particle log-likelihood, restarted from stored filtering particles (Li et al. 2023). L and the learning rate were chosen on the tuning year only (best L=10, lr=0.01, at the edge of the pre-planned grid).
- **Baselines:** Tam's 5-state UKF (μ_h from the tuning year, Gaussian predictive); bootstrap PF on the same SV-t model, frozen; GARCH(1,1) with normal or t errors, fitted once on the tuning year; GARCH(1,1)-t refitted every 20 days on all data to date.
- **Scores:** one-step-ahead predictive log score (primary) and CRPS; 5 seeds for particle methods; 95% moving-block bootstrap CIs (20-day blocks) on daily differences.

## Results (`results/e2/table_main.md`, `results/e2/fig_regime.png`)
| Model | log score / day | CRPS ×100 |
|---|---|---|
| GARCH(1,1)-t, refit every 20d | **2.385** | **1.277** |
| SV-t, online | 2.376 | 1.290 |
| SV-t + NN residual, online | 2.371 | 1.291 |
| Bootstrap PF, SV-t (frozen) | 2.318 | 1.320 |
| GARCH(1,1)-t (fit once) | 2.317 | 1.311 |
| SV-t + NN residual (frozen) | 2.305 | 1.328 |
| GARCH(1,1)-N (fit once) | 2.198 | 1.383 |
| UKF (5-state) | 2.196 | 1.472 |

1. **Online adaptation helps.** Online SV-t beats the same model frozen by +0.057/day [+0.032, +0.082] and the UKF by +0.179 [+0.148, +0.207].
2. **It does not beat a regularly refitted GARCH-t.** The difference is −0.009 [−0.021, +0.001]: statistically indistinguishable, with the point estimate favouring GARCH.
3. **The neural residual adds nothing and overfits.** Online: NN minus parametric −0.005 [−0.016, +0.006]. Frozen: −0.013 [−0.017, −0.010]. In-window minus next-day log score is larger with the NN (+0.014 [+0.006, +0.022]), most visibly around FTX (gap 0.098 vs 0.031; NN loses 0.095/day vs parametric in the following month). Its worst days are the extreme moves (FTX crash, February 2026 ±12–15%).
4. **Most of the UKF's deficit is its Gaussian predictive density.** It ties GARCH-N (+0.002 [−0.026, +0.033]) and trails every Student-t model.

## Reproduce
```bash
cd e2
../.venv/Scripts/python run_e2.py tune     # ~3 min; checkpointed per grid point
../.venv/Scripts/python run_e2.py main     # ~25 min CPU; checkpointed per (variant, seed)
../.venv/Scripts/python run_e2.py report   # baselines, table, figure
```
Overfitting-probe details: `results/e2/overfit_probe.json`. Tuning: `results/e2/tuning.json`.

## Limitations
One asset and one evaluation period; daily data only; L and lr chosen at the grid boundary; resampling gradients not differentiated (biased gradients, as in the reference implementation); GARCH refit cadence (20 days) was set a priori, not tuned.
