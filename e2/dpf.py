"""E2: stochastic-volatility differentiable particle filter with a small neural residual.

Model (daily demeaned log-returns r_t):
    h_t = mu + phi (h_{t-1} - mu) + g_theta(h_{t-1} - mu) + sigma_h eps_t,   eps_t ~ N(0, 1)
    r_t = exp(h_t / 2) * xi_t,   xi_t ~ standardised Student-t(nu) (unit variance)
g_theta is a 1-8-1 tanh MLP whose output layer is zero-initialised, so the model starts as
the parametric SV-t model; with use_nn=False it *is* that model (the bootstrap-PF baseline).

Filtering: bootstrap particle filter with reparameterised transitions and multinomial
resampling at every step; resampling indices are not differentiated (the "normal"
resampler setting of Li, Chen & Li 2023). Gradients flow through transitions and weights.

Online learning (following Li, Chen & Li 2023): after filtering r_t, maximise the
sliding-window log-likelihood estimate of r_{t-L+1:t}, restarting the window from the
stored filtering particles at t-L, with one Adam step per time step.
"""
import math

import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
LOG2PI = math.log(2 * math.pi)


def _inv_softplus(y):
    return math.log(math.expm1(y))


class SVModel(nn.Module):
    def __init__(self, mu0, phi0=0.95, sig0=0.2, nu0=6.0, hidden=8, use_nn=True):
        super().__init__()
        self.use_nn = use_nn
        self.mu = nn.Parameter(torch.tensor(float(mu0)))
        self.a = nn.Parameter(torch.tensor(math.atanh(phi0)))
        self.b = nn.Parameter(torch.tensor(_inv_softplus(sig0)))
        self.c = nn.Parameter(torch.tensor(_inv_softplus(nu0 - 2.0)))
        if use_nn:
            self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)

    @property
    def phi(self):
        return torch.tanh(self.a)

    @property
    def sig_h(self):
        return nn.functional.softplus(self.b)

    @property
    def nu(self):
        return 2.0 + nn.functional.softplus(self.c)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_particles(self, n, gen):
        sd = self.sig_h / torch.sqrt(1 - self.phi**2)
        return self.mu + sd * torch.randn(n, generator=gen)

    def transition(self, h, eps):
        x = h - self.mu
        drift = self.mu + self.phi * x
        if self.use_nn:
            drift = drift + self.net(x.unsqueeze(-1)).squeeze(-1)
        return drift + self.sig_h * eps

    def obs_logpdf(self, r, h):
        nu = self.nu
        scale = torch.exp(h / 2) * torch.sqrt((nu - 2) / nu)
        z = r / scale
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi)
                - torch.log(scale) - (nu + 1) / 2 * torch.log1p(z**2 / nu))

    def sample_obs(self, h, gen):
        nu = float(self.nu)
        t = torch.distributions.StudentT(torch.tensor(nu)).sample(h.shape)  # uses global RNG
        return torch.exp(h / 2) * math.sqrt((nu - 2) / nu) * t


def pf_window(model, r_seq, h0, gen):
    """Bootstrap PF over r_seq starting from equally weighted particles h0.
    Returns per-step log-likelihood increments (differentiable) and final particles."""
    n = h0.shape[0]
    h, incs = h0, []
    for r in r_seq:
        h = model.transition(h, torch.randn(n, generator=gen))
        logw = model.obs_logpdf(r, h)
        incs.append(torch.logsumexp(logw, 0) - math.log(n))
        idx = torch.multinomial(torch.softmax(logw.detach(), 0), n, replacement=True, generator=gen)
        h = h[idx]
    return torch.stack(incs), h


def pretrain(model, r, epochs=150, n=256, lr=0.01, seed=0):
    """Fit parameters on a segment by maximising the PF log-likelihood estimate."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rt = torch.as_tensor(r)
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        incs, _ = pf_window(model, rt, model.init_particles(n, gen), gen)
        loss = -incs.mean()
        loss.backward()
        opt.step()
        hist.append(-float(loss))
    return hist


@torch.no_grad()
def filter_through(model, r, n, gen):
    """Run the filter (no learning) through r; return final particles."""
    h = model.init_particles(n, gen)
    for x in torch.as_tensor(r):
        h = model.transition(h, torch.randn(n, generator=gen))
        logw = model.obs_logpdf(x, h)
        h = h[torch.multinomial(torch.softmax(logw, 0), n, replacement=True, generator=gen)]
    return h


def walk_forward(model, r, start, h_start, online, L=20, lr=3e-3, n_live=1000, n_train=256,
                 n_crps=2000, seed=0, crps=True):
    """One-step-ahead evaluation on r[start:], optionally updating parameters online.

    Returns dict of arrays over t = start..len(r)-1:
      pred_ll : log p(r_t | r_{<t}) estimated with parameters fitted up to t-1
      crps    : CRPS of the predictive mixture for r_t
      inwin   : mean in-window log-likelihood of the window ending at t under the parameters
                *after* the update at t (theta_t); NaN if not online. Overfitting probe:
                inwin[t] - pred_ll[t+1], both scored with theta_t.
    """
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr) if online else None
    rt = torch.as_tensor(r)
    hist = {start - 1: h_start.detach()}  # filtering particles after r_{t}
    out = {k: [] for k in ("pred_ll", "crps", "inwin")}
    h = h_start.detach()
    for t in range(start, len(r)):
        with torch.no_grad():
            hp = model.transition(h, torch.randn(n_live, generator=gen))
            logw = model.obs_logpdf(rt[t], hp)
            out["pred_ll"].append(float(torch.logsumexp(logw, 0) - math.log(n_live)))
            if crps:
                from common import crps_samples
                pick = hp[torch.randint(0, n_live, (n_crps,), generator=gen)]
                out["crps"].append(crps_samples(float(rt[t]), model.sample_obs(pick, gen).numpy()))
            h = hp[torch.multinomial(torch.softmax(logw, 0), n_live, replacement=True, generator=gen)]
            hist[t] = h
            hist.pop(t - L - 2, None)
        if online and (t - L) in hist:
            h0 = hist[t - L][torch.randint(0, n_live, (n_train,), generator=gen)]
            opt.zero_grad()
            incs, _ = pf_window(model, rt[t - L + 1:t + 1], h0, gen)
            loss = -incs.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            # in-window fit under the *updated* parameters (same start particles), so that
            # inwin[t] and pred_ll[t+1] are both scored with theta_t
            with torch.no_grad():
                incs_post, _ = pf_window(model, rt[t - L + 1:t + 1], h0, gen)
            out["inwin"].append(float(incs_post.mean()))
        else:
            out["inwin"].append(float("nan"))
    return {k: np.asarray(v) for k, v in out.items()}
