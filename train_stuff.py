# %%
import torch
from torch import Tensor, nn
import numpy as np
from numpy import ndarray
import scipy
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from neuralconstitutive.utils import beta
from neuralconstitutive.tipgeometry import TipGeometry, Conical, Spherical
from neuralconstitutive.torch.constitutive import (
    ConstitutiveEqn,
    PowerLawRheology,
    StandardLinearSolid,
)
from neuralconstitutive.indentation import Indentation, Triangular
from neuralconstitutive.dataset import IndentationDataset, split_app_ret
from neuralconstitutive.models import FullyConnectedNetwork, BernsteinNN
from neuralconstitutive.torch.ting import TingApproach

plr = PowerLawRheology(4.0, 0.42)
sls = StandardLinearSolid(8, 0.01, 2)
indent = Triangular(10.0, 0.2)
t = torch.linspace(0, 0.2, 100)
# tip = Conical(torch.pi / 18)
tip = Spherical(1)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
with torch.no_grad():
    ax.plot(t, plr(t), label="PLR")
    ax.plot(t, sls(t), label="SLS")
    ax.legend()


# %%
def F(
    t: Tensor, constit: ConstitutiveEqn, indent: Indentation, tip: TipGeometry
) -> Tensor:
    a, b = tip.alpha, tip.beta
    velocity, indent = indent.v_app, indent.i_app

    def _integrand(t_: ndarray, t: float) -> ndarray:
        with torch.no_grad():
            t_ = torch.tensor(t_).view(-1)
            dF = constit(t - t_) * velocity(t_) * indent(t_) ** (b - 1)
        return dF.numpy()

    f = torch.tensor([quad(_integrand, 0, t_i, args=(t_i,))[0] for t_i in t])
    return a * f


# %%
def find_t1(t: Tensor, constit: ConstitutiveEqn, indent: Indentation) -> Tensor:
    v_app, v_ret = indent.v_app, indent.v_ret
    t_max = indent.t_max

    def _integrand_ret(t_: ndarray, t: float) -> ndarray:
        with torch.no_grad():
            t_ = torch.tensor(t_).view(-1)
            out = constit(t - t_) * v_ret(t_)
        return out.numpy()

    def _integrand_app(t_: ndarray, t: float) -> ndarray:
        with torch.no_grad():
            t_ = torch.tensor(t_).view(-1)
            out = constit(t - t_) * v_app(t_)
        return out.numpy()

    def _equation(t1: ndarray, t: float) -> ndarray:
        I1 = quad(_integrand_app, t1, t_max, args=(t,))[0]
        I2 = quad(_integrand_ret, t_max, t, args=(t,))[0]
        return I1 + I2

    def _find_t1(t: float) -> float:
        if _equation(0.0, t) <= 0:
            return 0.0
        else:
            result = scipy.optimize.root_scalar(
                _equation, args=(t,), method="bisect", bracket=(0, t_max)
            )
            return result.root

    return torch.tensor([_find_t1(t_i) for t_i in tqdm(t)])


# %%
F_app = F(t, plr, indent, tip)
plt.plot(t, F_app)
# %%
t_ret = torch.linspace(0.2, 0.4, 100)
t1 = find_t1(t_ret, plr, indent)
# %%
plt.plot(t_ret, t1)
# %%
F_ret = F(t1, sls, indent, tip)
plt.plot(t_ret, F_ret)


# %%
def simulate_approach(constit: ConstitutiveEqn, tip: TipGeometry, v: float, t: Tensor):
    a, b = tip.alpha, tip.beta

    def _integrand(t_, t):
        with torch.no_grad():
            phi = constit(torch.tensor(t - t_)).numpy()
            return phi * v * (v * t_) ** (b - 1)

    f = torch.tensor([quad(_integrand, 0, t_i, args=(t_i,))[0] for t_i in t])
    return a * f


# %%
def approach_trapz(
    constit: ConstitutiveEqn, tip: TipGeometry, t: Tensor, v: Tensor, I: Tensor
):
    a, b = tip.alpha, tip.beta
    phi = constit(t)
    dI_beta = v * I ** (b - 1)

    def _inner(ind: int):
        # assuming that the data are equally spaced in time
        phi_ = torch.flip(phi[0 : ind + 1], dims=(0,))
        t_ = t[0 : ind + 1]
        dI_beta_ = dI_beta[0 : ind + 1]
        return torch.trapz(phi_ * dI_beta_, x=t_).view(-1)

    f = torch.cat([_inner(i) for i in range(len(t))], dim=0)
    return a * f


tip = Conical(torch.pi / 18)
t = torch.linspace(0, 0.2, 100)
with torch.no_grad():
    out = simulate_approach(sls, tip, 10, t)
    out2 = approach_trapz(sls, tip, t, 10 * torch.ones_like(t), 10 * t)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(t, out, label="Numerical Integration")
    ax.plot(t, out2, label="Trapz")
    ax.legend()
# %%
tip = Conical(torch.pi / 18)
t = torch.linspace(1e-2, 0.2, 200)
f = simulate_approach(sls, tip, 10, t)
dataset = IndentationDataset(t, 10 * t, 10 * torch.ones_like(t), f)
# %%
fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
t = dataset.time.view(-1)
yvals = (dataset.indent.view(-1), dataset.velocity.view(-1), dataset.force.view(-1))
ylabels = ("Indentation", "Velocity", "Force")
for ax, y, ylab in zip(axes, yvals, ylabels):
    ax.plot(t, y)
    ax.set_ylabel(ylab)
axes[-1].set_xlabel("Time")
# %%
dataset.force


# %%
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
ting_true = TingApproach(sls, tip)
# model = nn.Sequential(
#     nn.Linear(1, 20),
#     nn.GELU(),
#     nn.Linear(20, 20),
#     nn.GELU(),
#     nn.Linear(20, 20),
#     nn.GELU(),
#     nn.Linear(20, 1),
#     nn.Softplus(),
# )
model = FullyConnectedNetwork([1, 20, 20, 20, 1], torch.nn.functional.elu)
ting_nn = TingApproach(model, tip, lr=1e-3)
ting_bern = TingApproach(BernsteinNN(model, 100), tip, lr=1e-3)
with torch.no_grad():
    f_true = ting_true(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    f_nn = ting_nn(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    f_bern = ting_bern(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    axes[0].plot(t, f_true, label="Ground truth")
    axes[0].plot(t, f_nn, label="Untrained NN")
    axes[0].plot(t, f_bern, label="Untrained Bernstein NN")
    axes[0].legend()
    axes[0].set_ylabel("$F(t)$")

    phi_true = ting_true.stress_relaxation(t)
    phi_nn = ting_nn.stress_relaxation(t)
    phi_bern = ting_bern.stress_relaxation(t)
    axes[1].plot(t, phi_true)
    axes[1].plot(t, phi_nn)
    axes[1].plot(t, phi_bern)
    axes[1].legend()
    axes[1].set_ylabel("$\phi(t)$")
    axes[1].set_xlabel("$t$")
# %%
dataset.velocity.shape
# %%
f_bern
# %%
[model(t_.view(-1)) for t_ in dataset.time[0]]
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
trainer.fit(ting_bern, dataloader)
# %%
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
with torch.no_grad():
    f_true = ting_true(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    f_nn = ting_nn(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    f_bern = ting_bern(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    axes[0].plot(t, f_true, label="Ground truth")
    axes[0].plot(t, f_nn, label="Untrained NN")
    axes[0].plot(t, f_bern, label="Trained Bernstein NN")
    axes[0].legend()
    axes[0].set_ylabel("$F(t)$")

    phi_true = ting_true.stress_relaxation(t)
    phi_nn = ting_nn.stress_relaxation(t)
    phi_bern = ting_bern.stress_relaxation(t)
    axes[1].plot(t, phi_true)
    axes[1].plot(t, phi_nn)
    axes[1].plot(t, phi_bern)
    axes[1].legend()
    axes[1].set_ylabel("$\phi(t)$")
    axes[1].set_xlabel("$t$")
# %%
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
with torch.no_grad():
    f_bern = ting_bern(
        dataset.time.view(-1), dataset.velocity.view(-1), dataset.indent.view(-1)
    )
    axes[0].plot(dataset.time.view(-1), dataset.force.view(-1), label="Ground truth")
    axes[0].plot(dataset.time.view(-1), f_bern, label="Trained Bernstein NN")
    axes[0].legend()
    axes[0].set_ylabel("$F(t)$")

    phi_bern = ting_bern.stress_relaxation(dataset.time.view(-1))
    axes[1].plot(dataset.time.view(-1), phi_bern)
    axes[1].legend()
    axes[1].set_ylabel("$\phi(t)$")
    axes[1].set_xlabel("$t$")

# %%
with torch.no_grad():
    phi_start = ting_bern.stress_relaxation(dataset.time.view(-1))
    plt.plot(dataset.time.view(-1), phi_start)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
with torch.no_grad():
    # h = ting.model.func(dataset.time.view(-1, 1)).view(-1)
    # ax.plot(dataset.time.view(-1), h)
    t = torch.logspace(-4, 4, 10000, dtype=torch.float32)
    h = ting_bern.model.func(t.view(-1, 1)).view(-1)
    ax.plot(t, h)
# ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("$h(x)$")


# %%
def simulate_PLR(E0, gamma, t0, t_max, dt, tip):
    def t1(time):
        coeff = 2.0 ** (1.0 / (1.0 - gamma))
        return torch.clamp(time - coeff * (time - t_max), 0.0, None)

    a, b = torch.tensor(tip.alpha), torch.tensor(tip.beta)
    v = 10.0  # 10um/s
    gamma = torch.tensor(gamma)
    coeff = E0 * t0**gamma * a * b * v**b * beta(b, 1.0 - gamma)
    time = torch.arange(0.0, 2 * t_max, dt)
    is_app = time <= t_max
    time_1 = torch.cat([time[is_app], t1(time[~is_app])], axis=-1)
    force = coeff * time_1 ** (b - gamma)
    indent = torch.cat([v * time[is_app], 2 * v * t_max - v * time[~is_app]], axis=-1)
    return IndentationDataset(time.view(1, -1), indent.view(1, -1), force.view(1, -1))


tip = Conical(torch.pi / 10.0)
dataset = simulate_PLR(0.572, 0.42, 1.0, 0.2, 0.001, tip)

# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 4), constrained_layout=True)
axes[0].plot(dataset.indent[0], dataset.force[0])
axes[1].plot(dataset.time[0], dataset.indent[0])
axes[2].plot(dataset.time[0], dataset.force[0])


# %%
dataset_app, dataset_ret = split_app_ret(dataset)
# %%
print(dataset_app.time.shape)
# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 4), constrained_layout=True)
axes[0].plot(dataset_app.indent, dataset_app.force)
axes[1].plot(dataset_app.time, dataset_app.indent)
axes[2].plot(dataset_app.time, dataset_app.force)
axes[0].plot(dataset_ret.indent, dataset_ret.force)
axes[1].plot(dataset_ret.time, dataset_ret.indent)
axes[2].plot(dataset_ret.time, dataset_ret.force)

# %%
# logger = WandbLogger(project="hydrogel", entity="jhelab")
logger = None
model = FullyConnectedNetwork([1, 10, 10, 1], torch.nn.functional.relu)
ting = TingApproach(model, dataset_app.time, dataset_app.indent, lr=1e-3)
# %%
with torch.no_grad():
    f_pred = ting(dataset_app.time)
    print(f_pred.shape)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(dataset_app.time, dataset_app.force, label="Data")
axes[0].plot(dataset_app.time, f_pred, label="NN_untrained")
axes[0].legend()
with torch.no_grad():
    axes[1].plot(dataset_app.time, model(dataset_app.time.view(-1, 1)), ".")
# %%
dataloader = torch.utils.data.DataLoader(
    dataset_app,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
)
trainer = pl.Trainer(
    max_epochs=1000,
    log_every_n_steps=1,
    deterministic="warn",
    accelerator="gpu",
    devices=1,
    logger=logger,
)
trainer.fit(ting, dataloader)
# %%
with torch.no_grad():
    f_pred = ting(dataset_app.time.view(-1))
    print(f_pred.shape)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(dataset_app.time.view(-1), dataset_app.force.view(-1), label="Data")
axes[0].plot(dataset_app.time.view(-1), f_pred.view(-1), label="NN_untrained")
axes[0].legend()
with torch.no_grad():
    axes[1].plot(dataset_app.time.view(-1), model(dataset_app.time.view(-1, 1)), ".")
# %%
