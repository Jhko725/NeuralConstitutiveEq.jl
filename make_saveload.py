# %%
from typing import Callable

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

from neuralconstitutive.ting import TingApproach
from neuralconstitutive.tipgeometry import Conical
from neuralconstitutive.models import FullyConnectedNetwork, BernsteinNN
from neuralconstitutive.preprocessing import process_approach_data, estimate_derivative
from neuralconstitutive.dataset import IndentationDataset

ckpt_path = "hydrogel/w1cpp7ha/checkpoints/epoch=9999-step=10000.ckpt"
tip = Conical(torch.pi / 36)
model = FullyConnectedNetwork([1, 20, 20, 20, 1], torch.nn.functional.elu)
ting = TingApproach.load_from_checkpoint(
    checkpoint_path=ckpt_path, model=BernsteinNN(model, 100), tip=tip
)
# %%
datapath = "data/230602_highlyentangled_preliminary/Hydrogel(liquid, 10nN, 10s)-2.nid"
time, indent, force = process_approach_data(datapath, contact_point=1.64e-6, k=0.2)
velocity = estimate_derivative(time, indent)
dataset = IndentationDataset(
    time[1:],
    indent[1:] * 1e6,
    12.6 * np.ones_like(indent[1:]),
    force[1:] * 2e9,
    dtype=torch.float32,
)

# %%
with torch.no_grad():
    phi_start = ting.stress_relaxation(dataset.time.view(-1))
    plt.plot(dataset.time.view(-1), phi_start)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
with torch.no_grad():
    # h = ting.model.func(dataset.time.view(-1, 1)).view(-1)
    # ax.plot(dataset.time.view(-1), h)
    t = torch.logspace(-4, 4, 10000, dtype=torch.float32)
    h = ting.model.func(t.view(-1, 1)).view(-1)
    ax.plot(t, h)
# ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("$h(x)$")


# %%
def laplace_1d(t, func: Callable[[float], float]):
    def integrand(x):
        return np.exp(-t * x) * func(x)

    return scipy.integrate.quad(integrand, 0, np.inf, limit=100)


def func(x):
    with torch.no_grad():
        val = ting.model.func(torch.tensor(x, dtype=torch.float32).view(-1, 1)).view(-1)
    return val.numpy()


x, w = scipy.special.roots_laguerre(100)
x, w = np.float32(x), np.float32(w)


def gauss_laguerre(t, func: Callable[[float], float], x, w):
    f = func(x / t)
    return np.dot(w, f) / t


t_np = dataset.time.view(-1).numpy()
phi_scipy = np.array([laplace_1d(t_, func)[0] for t_ in t_np])
phi_laguerre = np.array([gauss_laguerre(t_, func, x, w) for t_ in t_np])
# %%
t_np.shape
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t_np, phi_scipy, ".", label="scipy.integrate.quad")
ax.plot(t_np, phi_laguerre, ".", label="gauss laguerre")
ax.legend()
ax.set_ylabel("$\int_0^\infty e^{-tx}h(x)dx$")
# %%
