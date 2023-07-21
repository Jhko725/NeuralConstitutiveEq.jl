# %%
from typing import Callable

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

from neuralconstitutive.ting import TingApproach
from neuralconstitutive.tipgeometry import Conical, Spherical
from neuralconstitutive.models import FullyConnectedNetwork, BernsteinNN
from neuralconstitutive.preprocessing import process_approach_data, estimate_derivative
from neuralconstitutive.dataset import IndentationDataset

ckpt_path = "hydrogel/h5hs8dpt/checkpoints/best_epoch=9998-step=9999.ckpt"
tip = Spherical(2.0) # diameter 0.8, 2.0, 4 -> R=0.4, 1.0, 2.0
model = FullyConnectedNetwork([1, 200, 1], torch.nn.functional.elu)
ting = TingApproach.load_from_checkpoint(
    checkpoint_path=ckpt_path, model=BernsteinNN(model, 100), tip=tip
)
# %%
datapath = "Hydrogel AFM data/SD-Sphere-CONT-L/Highly Entangled Hydrogel(10nN, 1s, liquid).nid"
time, indent, force = process_approach_data(datapath, contact_point=-1.2367037205320556e-06, k=0.2)
#time, indent, force = process_approach_data(datapath, contact_point=-1.7937788208936754e-06, k=0.2)
#time, indent, force = process_approach_data(datapath, contact_point=-1.592319017074137e-07, k=0.2)
velocity = estimate_derivative(time, indent)
dataset = IndentationDataset(
    time[1:] * 100,
    indent[1:] * 1e6,
    velocity[1:]*1e4,
    force[1:] * 2e9,
    dtype=torch.float32,
)
# %%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
with torch.no_grad():
    ting = ting.cpu()
    phi_start = ting.stress_relaxation(dataset.time.view(-1))
    ax.plot(dataset.time.view(-1)/100, phi_scipy/(2e9), ".")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$\phi(t)$")
    #ax.set_xscale("log", base=10)
    #ax.set_yscale("log", base = 10)
    ax.set_title("Setpoint = 10nN, Modulation time = 1s, Tip radius = 2.0um")
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
#%%
fig, axes = plt.subplots(1, 1, figsize=(5, 3), sharex=True)
with torch.no_grad():
    f_bern = ting(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    axes.plot(dataset.time.view(-1), dataset.force.view(-1), ".", label="Ground truth")
    axes.plot(dataset.time.view(-1), f_bern, ".", label="Trained Bernstein NN")
    axes.legend()
    axes.set_ylabel("$F(t)$")

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
