# %%
from typing import Callable
from functools import partial

import jax
from jax import config
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import jaxopt

config.update("jax_enable_x64", True)


class SimpleLinearSolid(eqx.Module):
    E0: float
    E_inf: float
    tau: float

    def __init__(self, E0: float, E_inf: float, tau: float):
        self.E0 = E0
        self.E_inf = E_inf
        self.tau = tau

    def __call__(self, t: jax.Array) -> jax.Array:
        return self.E_inf + (self.E0 - self.E_inf) * jnp.exp(-t / self.tau)


sls = SimpleLinearSolid(E0=8.0, E_inf=2.0, tau=0.01)
t_app = jnp.linspace(0, 0.2, 100)
t_ret = jnp.linspace(0.2, 0.4, 100)
sls(t_app)
v_app = 10.0 * jnp.ones_like(t_app)
v_ret = -10.0 * jnp.ones_like(t_ret)
# %%
grads = jax.grad(lambda t, model: model(t), argnums=1)(0.1, sls)
grads.E0


# %%
@jax.jit
def integrate_to(x_upper: float, x: Array, y: Array) -> Array:
    return jax.lax.cond(
        x_upper > x[0], _integrate_to, lambda *_: jnp.array(0.0), x_upper, x, y
    )


def _integrate_to(x_upper: float, x: Array, y: Array) -> Array:
    ind = jnp.searchsorted(x, x_upper)
    mask = jnp.arange(x.shape[0]) < ind
    y_, x_ = jnp.where(mask, y, 0), jnp.where(mask, x, x[ind - 1])
    y2, y1 = y[ind], y[ind - 1]
    x2, x1 = x[ind], x[ind - 1]
    y_upper = ((x_upper - x1) * y2 + (x2 - x_upper) * y1) / (x2 - x1)
    return jnp.trapz(y_, x=x_) + 0.5 * (y1 + y_upper) * (x_upper - x1)


@jax.jit
def integrate_from(x_lower: float, x: Array, y: Array) -> Array:
    return jax.lax.cond(
        x_lower < x[-1], _integrate_from, lambda *_: jnp.array(0.0), x_lower, x, y
    )


def _integrate_from(x_lower: float, x: Array, y: Array) -> Array:
    ind = jnp.searchsorted(x, x_lower, side="right")
    mask = jnp.arange(x.shape[0]) > ind
    y_, x_ = jnp.where(mask, y, 0), jnp.where(mask, x, x[ind])
    y2, y1 = y[ind], y[ind - 1]
    x2, x1 = x[ind], x[ind - 1]
    y_lower = ((x_lower - x1) * y2 + (x2 - x_lower) * y1) / (x2 - x1)
    return jnp.trapz(y_, x=x_) + 0.5 * (y1 + y_lower) * (x_lower - x1)


# %%
x = jnp.arange(10)
x_upper = 7.5
ind = jnp.searchsorted(x, x_upper)
mask = jnp.arange(x.shape[0]) < ind
print(x[ind], x[ind - 1])
x_ = jnp.where(mask, x, x[ind - 1])
x_
# %%
x = jnp.arange(10)
x_lower = 0.0
ind = jnp.searchsorted(x, x_lower, side="right")
mask = jnp.arange(x.shape[0]) >= ind
print(x[ind], x[ind - 1])
x_ = jnp.where(mask, x, x[ind])
x_
# %%
jax.grad(integrate_to)(0.25, t_ret, v_ret)
# %%
jax.grad(integrate_from)(0.2, t_app, v_app)
# %%
jnp.trapz(jnp.array([1, 2, 3]), x=jnp.array([1, 1, 2]))


# %%
def app_integral(t1, t, t_app, phi, v):
    inds = t_app >= t1
    t_, v_ = t_app[inds], v[inds]
    return jnp.trapz(phi(t - t_) * v_, x=t_)


def ret_integral(t, t_ret, phi, v):
    inds = t_ret <= t
    t_, v_ = t_ret[inds], v[inds]
    return jnp.trapz(phi(t - t_) * v_, x=t_)


def calc_t1(t, phi, t_app, t_ret, v_app, v_ret):
    def objective(t1):
        return app_integral(t1, t, t_app, phi, v_app) - ret_integral(
            t, t_ret, phi, v_ret
        )

    jax.lax.cond(objective(0.0) <= 0, jnp.ones_like)


print(ret_integral(0.3, t_ret, sls, v_ret))
print(app_integral(0.0, 0.3, t_app, sls, v_app))


# %%
def objective(
    t1: float,
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    phi_app, phi_ret = model(t - t_app), model(t - t_ret)
    return integrate_from(t1, t_app, phi_app * v_app) + integrate_to(
        t, t_ret, phi_ret * v_ret
    )


def find_t1(
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    return jax.lax.cond(
        objective(0.0, t, model, t_app, t_ret, v_app, v_ret) <= 0,
        lambda *_: jnp.array(0.0),
        _find_t1,
        t,
        model,
        t_app,
        t_ret,
        v_app,
        v_ret,
    )


def _find_t1(
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    root_finder = jaxopt.Bisection(optimality_fun=objective, lower=0.0, upper=0.2)
    return root_finder.run(
        t=t,
        model=model,
        t_app=t_app,
        t_ret=t_ret,
        v_app=v_app,
        v_ret=v_ret,
    ).params


objective(0.0, 0.3, sls, t_app, t_ret, v_app, v_ret)
# %%
_find_t1(0.3, sls, t_app, t_ret, v_app, v_ret)
# %%
t1_list = []
for t in t_ret:
    try:
        t1_list.append(_find_t1(t, sls, t_app, t_ret, v_app, v_ret))
    except ValueError:
        t1_list.append(0.0)
t1_list = jnp.array(t1_list)
# %%
t1_list
# %%
import matplotlib.pyplot as plt

plt.plot(t_ret, t1_list)
# %%
jax.grad(_find_t1, argnums=1)(0.3, sls, t_app, t_ret, v_app, v_ret).tau

# %%

# %%
import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from tqdm import tqdm


def PLR_relaxation(t, E0, gamma, t0):
    return E0 * ((t + 1e-7) / t0) ** (-gamma)


def indentation(t, v, t_max):
    return v * (t_max - jnp.abs(t - t_max))


# %%
@jax.jit
def cumtrapz(y, x):
    # y, x: 1D arrays
    dx = jnp.diff(x)
    y_mid = (y[0:-1] + y[1:]) / 2
    return jnp.insert(jnp.cumsum(y_mid * dx), 0, 0.0)


# Check if our cumtrapz implementation works
t = jnp.linspace(0, 2 * jnp.pi, 100)
y = jnp.cos(t)  # jnp.sin(t)
Y = cumtrapz(y, t)
Y.shape
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, y, label="original")
ax.plot(t, Y, label="Trapz")
ax.legend()
# %%
v = 10
t = jnp.linspace(0.0, 0.2, 100)
I = v * t
dI_beta = 2 * v * I  # 2IdI=2vI
phi = PLR_relaxation(t, 0.572, 0.42, 1.0)
phi_r = jnp.flip(phi)


def F(i):
    n = len(dI_beta)
    y = phi_r[n - i - 1 :] * dI_beta[: i + 1]
    return jnp.trapz(y, x=t[: i + 1])


force = [F(i) for i in tqdm(range(len(dI_beta)))]
# %%
plt.plot(t, force)


# %%
def d_force(t_, u, args):
    E0, gamma, t0, v, t_max, t = args
    phi = PLR_relaxation(t - t_, E0, gamma, t0)
    df = phi * 2 * v * jnp.sign(t_max - t_) * indentation(t_, v, t_max)
    return df


def force_true(t, args):
    E0, gamma, t0, v, t_max = args
    b = 2.0
    coeff = (
        E0
        * t0**gamma
        * b
        * v**b
        * jnp.exp(jax.scipy.special.betaln(b, 1.0 - gamma))
    )
    return coeff * t ** (b - gamma)


args = (0.572, 0.42, 1.0, 10, 0.2, 0.1)
d_force(0.19, jnp.array([0.0]), args)


# %%
def force_trapz(t, ts):
    args = (0.572, 0.42, 1.0, 10, 0.2, t)
    ts_ = ts[ts <= t]
    df = d_force(ts_, None, args)
    return jnp.trapz(df, x=ts_)


ts = jnp.linspace(0, 0.199, 100)
out_trapz = jnp.stack([force_trapz(t, ts) for t in tqdm(ts)], axis=-1)
out_trapz
# %%
out_trapz
# %%
solver = diffrax.Midpoint()
term = diffrax.ODETerm(d_force)
sol = diffrax.diffeqsolve(term, solver, 0.0, 0.21, 0.1, jnp.array([0.0]), args=args)
# %%
sol.ys
# %%
ts = jnp.linspace(0, 0.2, 100)
out = jnp.concatenate(
    [
        diffrax.diffeqsolve(
            term,
            solver,
            0.0,
            t,
            0.01,
            jnp.array([0.0]),
            args=(0.572, 0.42, 1.0, 10, 0.2, t),
        ).ys
        for t in ts
    ],
    axis=-1,
)
# %%
out
# %%
out_true = force_true(ts, (0.572, 0.42, 1.0, 10, 0.2))
# %%
out_true.shape
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(ts, out[0], label="ODESolve", alpha=0.5)
ax.plot(ts, force, label="trapz", alpha=0.5)
ax.plot(ts, out_true, label="analytical", alpha=0.5)
ax.legend()
# %%
# %%
