# %%
from typing import Callable
import torch
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    ConstitutiveEqn,
    PowerLawRheology,
    StandardLinearSolid,
)

sls = StandardLinearSolid(8, 0.05)
# t = torch.linspace(0, 0.4, 200)
# t_app, t_ret = t[t <= 0.2], t[t > 0.2]
t_app = torch.linspace(0, 0.2, 100)
t_ret = torch.linspace(0.2, 0.4, 100)


# %%
def find_t1(t: float, phi: Callable, v: float, t_max: float):
    def _inner(t1):
        term1 = scipy.integrate.quad(lambda t_: phi(t - t_) * v, t1, t_max)[0]
        term2 = scipy.integrate.quad(lambda t_: -phi(t - t_) * v, t_max, t)[0]
        return term1 + term2

    if _inner(0.0) <= 0:
        return 0.0
    else:
        return scipy.optimize.root_scalar(
            _inner, method="bisect", bracket=(0.0, t_max)
        ).root


def phi(t):
    with torch.no_grad():
        return sls(torch.tensor(t).view(-1)).numpy()


# %%
t1 = np.array([find_t1(t_, phi, 10.0, 0.2) for t_ in tqdm(t_ret)])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_ret.numpy(), t1, label="scipy")
ax.legend()


# %%
def ret_integral(t, t_ret, v_ret, phi):
    t_ret_ = t_ret[t_ret <= t]
    v_ret_ = v_ret[t_ret <= t]
    phi_ = phi(t - t_ret_)
    return torch.trapz(phi_ * v_ret_, x=t_ret_)


ret_val = ret_integral(0.22, t_ret, 10 * torch.ones_like(t_ret), sls)


# %%
def app_integral(t, t_app, v_app, phi):
    phi_ = phi(t - torch.flip(t_app, dims=(0,)))
    return torch.cat(
        [
            torch.trapz(phi_[0 : i + 1] * v_app[0 : i + 1], x=t_app[0 : i + 1]).view(-1)
            for i in range(len(phi_))
        ],
        dim=0,
    )


app_vals = app_integral(0.22, t_app, 10 * torch.ones_like(t_app), sls)
# %%
ind = torch.searchsorted(app_vals, ret_val)
t_app[len(t_app) - ind]
# %%
t1_torch = []
inds = []
with torch.no_grad():
    for t_ in tqdm(t_ret):
        ret_val = ret_integral(t_, t_ret, 10 * torch.ones_like(t_ret), sls)
        app_vals = app_integral(t_, t_app, 10 * torch.ones_like(t_app), sls)
        ind = torch.searchsorted(app_vals, ret_val) - 1
        t1_torch.append(t_app[-1] - t_app[torch.clamp(ind, min=0)])
        inds.append(ind)
t1_torch = np.array(t1_torch)

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_ret.numpy(), t1, ".", label="scipy")
ax.plot(t_ret.numpy(), t1_torch, ".", label="torch")
ax.legend()
ax.set_xlabel("$t$ (retraction)")
ax.set_ylabel("$t_1$")
# %%
inds
# %%
