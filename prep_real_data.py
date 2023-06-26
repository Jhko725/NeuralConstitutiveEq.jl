# %%
from typing import Callable
import numpy as np
import scipy
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt

from neuralconstitutive.tipgeometry import Conical
from neuralconstitutive.models import BernsteinNN, FullyConnectedNetwork
from neuralconstitutive.ting import TingApproach
from neuralconstitutive.dataset import IndentationDataset
from neuralconstitutive.preprocessing import process_approach_data, estimate_derivative

datapath = "data/230602_highlyentangled_preliminary/Hydrogel(liquid, 50nN, 10s).nid"

# %%
time, indent, force = process_approach_data(datapath, contact_point=1.64e-6, k=0.2)
velocity = estimate_derivative(time, indent)
fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
yvals = (indent, velocity, force)
ylabels = ("Indentation (m)", "Velocity (m/s)", "Force (N)")
for ax, y, ylab in zip(axes, yvals, ylabels):
    ax.plot(time, y)
    ax.set_ylabel(ylab)
axes[-1].set_xlabel("Time (s)")


# %%
tip = Conical(torch.pi / 36)
# model = nn.Sequential(
#     nn.Linear(1, 20),
#     nn.ELU(),
#     nn.Linear(20, 20),
#     nn.ELU(),
#     nn.Linear(20, 20),
#     nn.ELU(),
#     nn.Linear(20, 1),
# )
model = FullyConnectedNetwork([1, 200, 1], torch.nn.functional.elu)
ting = TingApproach(BernsteinNN(model, 100), tip, lr=1e-2)
# dataset = IndentationDataset(
#     time[1:] / 10,
#     indent[1:] * 1e6,
#     velocity[1:] * 1e6 * 10,
#     force[1:] * 1e9,
#     dtype=torch.float32,
# )

dataset = IndentationDataset(
    time[1:] * 100,
    indent[1:] * 1e6,
    0.126 * np.ones_like(indent[1:]),
    force[1:] * 2e9,
    dtype=torch.float32,
)
# %%
dataset.time
# %%
# ting = ting.double()
# %%
fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
yvals = (dataset.indent.view(-1), dataset.velocity.view(-1), dataset.force.view(-1))
ylabels = ("Indentation", "Velocity", "Force")
for ax, y, ylab in zip(axes, yvals, ylabels):
    ax.plot(dataset.time.view(-1), y)
    ax.set_ylabel(ylab)
axes[-1].set_xlabel("Time")

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
print(ting.model.scale, ting.model.offset)
# %%
logger = WandbLogger(project="hydrogel", entity="jhelab")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
)
trainer = pl.Trainer(
    max_epochs=10000,
    log_every_n_steps=1,
    deterministic="warn",
    accelerator="gpu",
    devices=1,
    logger=logger,
)
trainer.fit(ting, dataloader)

# %%
ting = ting.cpu()
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
with torch.no_grad():
    f_bern = ting(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    axes[0].plot(dataset.time.view(-1), dataset.force.view(-1), label="Ground truth")
    axes[0].plot(dataset.time.view(-1), f_bern, label="Trained Bernstein NN")
    axes[0].legend()
    axes[0].set_ylabel("$F(t)$")

    phi_bern = ting.stress_relaxation(dataset.time.view(-1))
    axes[1].plot(dataset.time.view(-1), phi_bern)
    axes[1].legend()
    axes[1].set_ylabel("$\phi(t)$")
    axes[1].set_xlabel("$t$")


# %%
# %%
def laplace_1d(t, func: Callable[[float], float]):
    def integrand(x):
        return np.exp(-t * x) * func(x)

    return scipy.integrate.quad(integrand, 0, np.inf, limit=100)


def func(x):
    with torch.no_grad():
        val = ting.model.func(torch.tensor(x, dtype=torch.float32).view(-1, 1)).view(-1)
    return val.numpy()


x, w = scipy.special.roots_laguerre(150)
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
