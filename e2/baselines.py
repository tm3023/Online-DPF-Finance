"""E2 baselines: Tam's 5-state UKF and GARCH(1,1), scored one step ahead on returns.

UKF: replica of run_base_filter from github.com/tm3023/Bitcoin-UKF (btc_modal.py), with
one change: the long-run log-variance mu_h is set from the tuning segment rather than the
full sample (the repo's 2*log(r.std()) uses the whole series, a mild look-ahead).
The one-step predictive density of r_t is N(z_pred[0], S[0,0]) from the UKF prior.
"""
import numpy as np
from arch import arch_model
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy import stats

from common import crps_gaussian, crps_samples


def _damped_osc(omega, zeta, dt=1.0):
    wd = omega * np.sqrt(max(1 - zeta**2, 1e-8))
    e = np.exp(-zeta * omega * dt)
    c, s = np.cos(wd * dt), np.sin(wd * dt)
    return e * np.array([[c + zeta * omega / wd * s, s / wd],
                         [-(omega**2) / wd * s, c - zeta * omega / wd * s]])


def ukf_scores(r, n_tune):
    mu_h = 2 * np.log(r[:n_tune].std())
    phi = 0.95
    A1, A2 = _damped_osc(2 * np.pi / 30.0, 0.15), _damped_osc(2 * np.pi / 5.0, 0.35)

    def fx(x, dt):
        p1, v1, p2, v2, h = x
        a, b = A1 @ [p1, v1], A2 @ [p2, v2]
        return np.array([a[0], a[1], b[0], b[1], mu_h + phi * (h - mu_h)])

    def hx(x):
        return np.array([x[0] + x[2], x[4] - 1.27])

    ukf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=1.0, fx=fx, hx=hx,
                                points=MerweScaledSigmaPoints(5, alpha=1e-3, beta=2.0, kappa=0.0))
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, mu_h])
    ukf.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    ukf.Q = np.diag([1e-6, 1e-5, 1e-5, 1e-4, 0.05])
    n = len(r)
    ll, crps = np.empty(n), np.empty(n)
    for t in range(n):
        ukf.R = np.array([[np.exp(ukf.x[4]), 0.0], [0.0, np.pi**2 / 2.0]])
        ukf.predict()
        z = np.array([r[t], np.log(r[t]**2 + 1e-8)])
        ukf.update(z)
        mu, var = z[0] - ukf.y[0], ukf.S[0, 0]   # prior predictive mean / variance of r_t
        ll[t] = stats.norm.logpdf(r[t], mu, np.sqrt(var))
        crps[t] = crps_gaussian(r[t], mu, np.sqrt(var))
    return ll, crps


def garch_scores(r, n_tune, dist="normal", n_samp=4000, seed=0):
    """GARCH(1,1) fitted on the tuning segment only, then run forward with fixed parameters."""
    rng = np.random.default_rng(seed)
    x = 100 * r
    res = arch_model(x[:n_tune], mean="Constant", vol="GARCH", p=1, q=1, dist=dist).fit(disp="off")
    p = res.params
    mu, om, al, be = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
    nu = p.get("nu", None)
    n = len(r)
    s2 = np.empty(n)
    s2[0] = x[:n_tune].var()
    for t in range(1, n):
        s2[t] = om + al * (x[t - 1] - mu) ** 2 + be * s2[t - 1]
    sig = np.sqrt(s2) / 100
    m = mu / 100
    ll, crps = np.empty(n), np.empty(n)
    if dist == "normal":
        ll = stats.norm.logpdf(r, m, sig)
        crps = crps_gaussian(r, m, sig)
    else:  # standardised Student-t: scale = sigma * sqrt((nu-2)/nu)
        sc = sig * np.sqrt((nu - 2) / nu)
        ll = stats.t.logpdf(r, nu, m, sc)
        z = rng.standard_t(nu, size=n_samp)
        crps = np.array([crps_samples(r[t], m + sc[t] * z) for t in range(n)])
    return ll, crps, {k: float(v) for k, v in p.items()}


def garch_t_rolling_scores(r, n_tune, refit_every=20, n_samp=4000, seed=0):
    """GARCH(1,1)-t refitted every `refit_every` days on all data up to the refit date
    (expanding window, no look-ahead); the variance recursion runs on through each block."""
    rng = np.random.default_rng(seed)
    x = 100 * r
    n = len(r)
    ll, crps = np.full(n, np.nan), np.full(n, np.nan)
    z = rng.standard_t(1, size=1)  # placeholder, replaced per refit
    params_log = []
    for start in range(n_tune, n, refit_every):
        res = arch_model(x[:start], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        p = res.params
        mu, om, al, be, nu = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"], p["nu"]
        params_log.append({"start": int(start), **{k: float(v) for k, v in p.items()}})
        s2 = x[:start].var()
        for t in range(1, start + 1):  # filter the variance up to the block start with current params
            s2 = om + al * (x[t - 1] - mu) ** 2 + be * s2
        z = rng.standard_t(nu, size=n_samp)
        for t in range(start, min(start + refit_every, n)):
            if t > start:
                s2 = om + al * (x[t - 1] - mu) ** 2 + be * s2
            sc = np.sqrt(s2) / 100 * np.sqrt((nu - 2) / nu)
            ll[t] = stats.t.logpdf(r[t], nu, mu / 100, sc)
            crps[t] = crps_samples(r[t], mu / 100 + sc * z)
    return ll, crps, params_log
