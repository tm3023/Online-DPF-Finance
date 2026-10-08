# E2: BTC-USD one-step-ahead density forecasts (evaluation 2022-10-02 – 2026-10-02, 1462 days)

Fit/tuning segment: 2021-10-03 – 2022-10-01 (364 days); returns demeaned with its mean (-0.00248). Online window L=10, Adam lr=0.01 (chosen on the tuning segment only). Parameters: SV-t 4, SV-t+NN 29. Seeds: [0, 1, 2, 3, 4]. GARCH baselines: fitted once on the tuning year, or refitted every 20 days on all data to date (expanding window, no look-ahead).

| Model | mean log score / day (± sd over seeds) | CRPS ×100 | Δ vs bootstrap PF [95% CI] | Δ vs UKF [95% CI] | Δ vs rolling GARCH-t [95% CI] |
|---|---|---|---|---|---|
| GARCH(1,1)-t, refit every 20d | 2.3849 ± 0.0000 | 1.2770 | +0.0665 [+0.0411, +0.0913] | +0.1886 [+0.1573, +0.2175] | +0.0000 [+0.0000, +0.0000] |
| SV-t, online | 2.3756 ± 0.0004 | 1.2897 | +0.0573 [+0.0315, +0.0815] | +0.1794 [+0.1477, +0.2074] | -0.0092 [-0.0209, +0.0014] |
| SV-t + NN residual, online | 2.3711 ± 0.0014 | 1.2913 | +0.0527 [+0.0209, +0.0826] | +0.1748 [+0.1385, +0.2072] | -0.0138 [-0.0313, +0.0025] |
| Bootstrap PF, SV-t (frozen) | 2.3183 ± 0.0017 | 1.3196 | +0.0000 [+0.0000, +0.0000] | +0.1221 [+0.0976, +0.1482] | -0.0665 [-0.0913, -0.0411] |
| GARCH(1,1)-t | 2.3168 ± 0.0000 | 1.3111 | -0.0015 [-0.0043, +0.0016] | +0.1206 [+0.0967, +0.1466] | -0.0680 [-0.0939, -0.0411] |
| SV-t + NN residual (frozen) | 2.3051 ± 0.0024 | 1.3281 | -0.0132 [-0.0167, -0.0095] | +0.1089 [+0.0834, +0.1361] | -0.0797 [-0.1073, -0.0515] |
| GARCH(1,1)-N | 2.1981 ± 0.0000 | 1.3834 | -0.1202 [-0.1386, -0.0997] | +0.0019 [-0.0260, +0.0325] | -0.1868 [-0.2266, -0.1449] |
| UKF (Tam, 5-state) | 2.1962 ± 0.0000 | 1.4724 | -0.1221 [-0.1482, -0.0976] | +0.0000 [+0.0000, +0.0000] | -0.1886 [-0.2175, -0.1573] |

Higher log score and lower CRPS are better. CIs: moving-block bootstrap (block 20 days) on the daily difference (DPF variants averaged over seeds first).

- Overfitting probe, SV-t + NN residual, online: mean in-window − next-day log score = 0.0166 (seed sd 0.0011)
- Overfitting probe, SV-t, online: mean in-window − next-day log score = 0.0027 (seed sd 0.0004)

Final parameters (seed 0): Bootstrap PF, SV-t (frozen): mu=-6.708, phi=0.820, sig_h=0.188, nu=3.848; SV-t + NN residual (frozen): mu=-6.806, phi=0.887, sig_h=0.564, nu=4.498; SV-t, online: mu=-7.741, phi=0.762, sig_h=0.251, nu=3.418; SV-t + NN residual, online: mu=-7.421, phi=0.598, sig_h=0.686, nu=4.141

GARCH parameters (tuning fit, % returns): GARCH(1,1)-N: mu=0.0129, omega=2.2239, alpha[1]=0.0073, beta[1]=0.8056; GARCH(1,1)-t: mu=0.1416, omega=1.7165, alpha[1]=0.0112, beta[1]=0.8612, nu=3.4767; GARCH(1,1)-t rolling (last refit): mu=0.2782, omega=0.1923, alpha[1]=0.0703, beta[1]=0.9226, nu=3.0611
