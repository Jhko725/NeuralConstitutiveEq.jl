# %%
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import jaxopt

def SLS(t: jax.Array, E0: float, E_inf: float, tau: float) -> jax.Array:
    return E_inf+(E0-E_inf)*jnp.exp(-t/tau)

sls = partial(SLS, E0=8.0, E_inf = 2.0, tau = 0.01)
t_app = jnp.linspace(0, 0.2, 100)
t_ret = jnp.linspace(0.2, 0.4, 100)
sls(t_app)
v_app = 10.0*jnp.ones_like(t_app)
v_ret = -10.0*jnp.ones_like(t_ret)
#%%
@jax.jit
def integrate_to(y: Array, x: Array, x_upper: float) -> Array:
    return jax.lax.cond(x_upper>x[0], _integrate_to, lambda *_: jnp.array(0.0), y, x, x_upper)
    
def _integrate_to(y: Array, x: Array, x_upper: float) -> Array:
    ind = jnp.searchsorted(x, x_upper)
    mask = jnp.arange(x.shape[0]) < ind
    y_, x_ = jnp.where(mask, y, 0), jnp.where(mask, x, 0)
    y2, y1 = y[ind], y[ind-1]
    x2, x1 = x[ind], x[ind-1]
    y_upper = ((x_upper-x1)*y2+(x2-x_upper)*y1)/(x2-x1)
    return jnp.trapz(y_, x=x_)+0.5*(y1+y_upper)*(x_upper-x1)

def integrate_from(y: Array, x: Array, x_lower: float) -> Array:
    ind = jnp.searchsorted(x, x_lower)
    mask = jnp.arange(x.shape[0]) > ind
    y_, x_ = jnp.where(mask, y, 0), jnp.where(mask, x, 0)
    y2, y1 = y[ind], y[ind-1]
    x2, x1 = x[ind], x[ind-1]
#%%
jax.grad(integrate_to, argnums=2)(v_ret, t_ret, 0.19)
#%%
def app_integral(t1, t, t_app, phi, v):
    inds = t_app >= t1
    t_, v_ = t_app[inds], v[inds]
    return jnp.trapz(phi(t-t_)*v_, x=t_)

def ret_integral(t, t_ret, phi, v):
    inds = t_ret <= t
    t_, v_ = t_ret[inds], v[inds]
    return jnp.trapz(phi(t-t_)*v_, x=t_)

def calc_t1(t, phi, t_app, t_ret, v_app, v_ret):
    def objective(t1):
        return app_integral(t1, t, t_app, phi, v_app) - ret_integral(t, t_ret, phi, v_ret)
    jax.lax.cond(objective(0.0)<=0, jnp.ones_like)
print(ret_integral(0.3, t_ret, sls, v_ret))
print(app_integral(0.0, 0.3, t_app, sls, v_app))
# %%
t1_list = []
for t in t_ret:
    try:
        bisect = jaxopt.Bisection(lambda t1: app_integral(t1, t, t_app, sls, v_app)+ret_integral(t, t_ret, sls, v_ret), lower=0.0, upper=0.2, jit=False)
        t1_list.append(bisect.run().params)
    except ValueError:
        t1_list.append(0.0)
t1_list = jnp.array(t1_list)
#%%
t1_list
#%%
import matplotlib.pyplot as plt
plt.plot(t_ret, t1_list)

#%%
def t1_from_params(t, E0, E_inf, tau, t_app, t_ret, v_app, v_ret):
    
    def objective(t1, t, E0, E_inf, tau):
        sls = partial(SLS, E0=E0, E_inf = E_inf, tau = tau)
        return app_integral(t1, t, t_app, sls, v_app)+ret_integral(t, t_ret, sls, v_ret)
    bisect = jaxopt.Bisection(objective, lower=0.0, upper=0.2, jit=False)
    return bisect.run(t=t, E0=E0, E_inf=E_inf, tau=tau).params

jax.grad(t1_from_params, argnums=2)(0.21, 8.0, 2.0, 0.01, t_app, t_ret, v_app, v_ret)

#%%
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
