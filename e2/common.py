"""E2 shared pieces: data, split, scoring rules."""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "btc_usd_close_2021-10-02_2026-10-02.csv")  # same file as E0 (sha256 c60148f4…)
OUT = os.path.join(ROOT, "results", "e2")
TUNE_END = "2022-10-02"  # returns dated before this form the tuning/fitting segment (~1 year)

# Stress episodes used only for plot annotation (same list as E0)
EVENTS = {"2022-11-08": "FTX", "2023-03-10": "SVB/USDC", "2024-08-05": "Yen carry unwind"}


def load_returns():
    s = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:, 0]
    r = np.diff(np.log(s.values.astype(float)))
    dates = s.index[1:]
    n_tune = int(np.sum(dates < pd.Timestamp(TUNE_END)))
    mean_tune = float(r[:n_tune].mean())
    return r - mean_tune, dates, n_tune, mean_tune  # demeaned with the tuning-segment mean only


def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_samples(y, x):
    """CRPS from predictive samples x (1-D) via the sorted-sample identity."""
    x = np.sort(np.asarray(x, dtype=float))
    m = len(x)
    term1 = np.mean(np.abs(x - y))
    term2 = np.sum((2 * np.arange(1, m + 1) - m - 1) * x) / (m * m)  # = 0.5 E|X - X'|
    return term1 - term2


def block_bootstrap_ci(d, block=20, n_boot=4000, seed=0):
    """Moving-block bootstrap 95% CI for the mean of a (serially dependent) series d."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    n = len(d)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return float(np.mean(d)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
