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
