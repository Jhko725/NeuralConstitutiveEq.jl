# %%
from typing import Callable
import functools

import numpy as np
import scipy
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt


def laplace_1d(t, func: Callable[[float], float]):
    def integrand(x):
        return np.exp(-t * x) * func(x)

    return quad(integrand, 0, np.inf)


def gauss_laguerre(t, func: Callable[[float], float], n: int = 100):
    x, w = scipy.special.roots_laguerre(n)
    f = func(x / t)
    return np.dot(w, f) / t


def gauss_laguerre2(t, func: Callable[[float], float], x, w):
    f = func(x / t)
    return np.dot(w, f) / t


def func(x):
    # Can be changed to be any positive function
    return x


x, w = scipy.special.roots_laguerre(100)
t = np.linspace(0.1, 3.0, 100)
# %%

out = np.array([laplace_1d(t_, func)[0] for t_ in t])
# %%

out2 = np.array([gauss_laguerre(t_, func, 100) for t_ in t])
# %%

out2 = np.array([gauss_laguerre2(t_, func, x, w) for t_ in t])
# %%
plot_kwargs = {"alpha": 0.9, "markersize": 3.0}
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, out, ".", label="scipy.integrad.quad", **plot_kwargs)
ax.plot(t, out2, ".", label="gauss_laguerre", **plot_kwargs)
ax.set_xlabel("$t$")
ax.set_ylabel("$\phi(t)$")
ax.legend()


# %%

t = np.linspace(0.1, 3.0, 100)
out2 = np.array([gauss_laguerre(t_, func) for t_ in tqdm(t)])
# %%
plt.plot(t, out2)
# %%
out[0]
# %%
import torch
from torch import nn


class BernsteinNN(nn.Module):
    def __init__(self, func: Callable, n: int):
        super().__init__()
        x, w = scipy.special.roots_laguerre(n)
        self.register_buffer(
            "nodes", torch.tensor(x, dtype=torch.float32).view(-1, 1)
        )  # shape: (n, 1)
        self.register_buffer(
            "weights", torch.tensor(w, dtype=torch.float32)
        )  # shape: (n,)
        self.func = func  # should map (n, 1) -> (n, 1)

    def forward(self, t):  # t has shape (d,); b: batch size
        x = self.nodes / t  # (n, d)
        f = self.func(x.view(-1, 1)).view(x.shape)  # (n, d) -> (nd, 1) -> (n, d)
        y = torch.mv(f.T, self.weights)  # (d, n)*(n,) -> (d,)
        return y / t


model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.ReLU(),
)
bernstein = BernsteinNN(model, 100)

t = torch.linspace(0.1, 3.0, 100)
with torch.no_grad():
    y_nn = model(t.view(-1, 1)).view(-1)
    y_bern = bernstein(t)
# %%
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
axes[0].plot(t, y_nn, label="Neural Network (naive)")
axes[1].plot(t, y_bern, label="Bernstein Neural Network")
axes[1].set_xlabel("$t$")
# %%
y_bern = bernstein(t)
# %%
