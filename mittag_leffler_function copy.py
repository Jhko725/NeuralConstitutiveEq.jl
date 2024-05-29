# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import gamma
from quadax import quadgk
from tqdm import tqdm

from mittag_leffler_master.mittag_leffler import ml

jax.config.update("jax_enable_x64", True)


# %%
def MLF1(alpha, beta, z, rho=1e-10):
    a = jnp.ceil((1 - beta) / alpha)
    b = jnp.ceil(jnp.log(rho * (1 - jnp.abs(z))) / jnp.log(jnp.abs(z)))
    k0 = jax.lax.max(a, b)
    k = jnp.arange(k0 + 1)
    return jnp.sum(z**k / gamma(beta + alpha * k))


def MLF2(alpha, beta, z, rho=1e-10):
    k0 = jnp.floor(-jnp.log(rho) / jnp.log(jnp.abs(z)))
    z_inv_alpha = z ** (1 / alpha)
    val = 0.0  # z_inv_alpha ** (1 - beta) * jnp.exp(z_inv_alpha) / alpha
    k = jnp.arange(1, k0 + 1)
    val = val - jnp.sum(z ** (-k) / gamma(beta - alpha * k))
    return val


def K(alpha, beta, chi, z):
    return (
        chi ** ((1 - beta) / alpha)
        / (alpha * jnp.pi)
        * jnp.exp(-(chi ** (1 / alpha)))
        * (
            chi * jnp.sin(jnp.pi * (1 - beta))
            - z * jnp.sin(jnp.pi * (1 - beta + alpha))
        )
    )


def P(alpha, beta, phi, z):
    eps = jnp.abs(z) / 2
    w = phi * (1 + (1 - beta) / alpha) + eps ** (1 / alpha) * jnp.sin(phi / alpha)
    return (
        eps ** (1 + (1 - beta) / alpha)
        / (2 * alpha * jnp.pi)
        * jnp.exp(eps ** (1 / alpha) * jnp.cos(phi / alpha))
        * ((complex(jnp.cos(w), jnp.sin(w))) / ((eps * jnp.exp(complex(0, phi))) - z))
    )


def MLF3(alpha, beta, z, rho=1e-5):
    if beta < 0:
        chi0 = max(
            (jnp.abs(beta) + 1) ** alpha,
            2 * jnp.abs(z),
            (
                -2
                * jnp.log(
                    jnp.pi
                    * rho
                    / (
                        6
                        * (jnp.abs(beta) + 2)
                        * ((2 * jnp.abs(beta)) ** (jnp.abs(beta)))
                    )
                )
                ** alpha
            ),
        )
    else:
        chi0 = max((1, 2 * jnp.abs(z), (-jnp.log(jnp.pi * rho / 6)) ** alpha))
    val = (
        quadgk(K, [0, chi0], args=(alpha, beta, z))[0]
        + z ** ((1 - beta) / alpha) * jnp.exp(z ** (1 / alpha)) / alpha
    )
    return val


def MLF4(alpha, beta, z, rho=1e-5):
    chi0 = max((1, 2 * jnp.abs(z), (-jnp.log(jnp.pi * rho / 6)) ** alpha))
    val = quadgk(K, [jnp.abs(z) / 2, chi0], args=(alpha, beta, z))[0]
    +quadgk(P, [-alpha * jnp.pi, alpha * jnp.pi], args=(alpha, beta, z))
    +(z ** ((1 - beta) / alpha)) * jnp.exp(z ** (1 / alpha)) / alpha
    return val


# %%
def MLF(alpha, beta, z) -> Array:
    alpha, beta, z = promote_args_inexact("MLF", alpha, beta, z)
    _c = _lax_const
    zero = _c(z, 0)
    one = _c(z, 1)
    conds = [
        (alpha <= 0) | (alpha > 1),
        (0 < alpha <= 1) & (z == 0),
        (0 < alpha <= 1) & (jnp.abs(z) < 1),
        (0 < alpha <= 1) & (jnp.abs(z) > jnp.floor(10 + 5 * alpha)),
        (0 < alpha <= 1) & (1 <= jnp.abs(z) <= jnp.floor(10 + 5 * alpha)) & (beta <= 1),
        # (0<alpha<=1) & (1<=jnp.abs(z)<=jnp.floor(10+5*alpha)) & (1<beta),
    ]
    vals = [
        jnp.nan,
        gamma(beta),
        MLF1(alpha, beta, z),
        MLF2(alpha, beta, z),
        MLF3(alpha, beta, z),
        # MLF4(alpha, beta, z)
    ]
    ret = jnp.piecewise(z, conds, vals)
    return ret


# %%
### condition : (0<alpha<=1) & (jnp.abs(z)<1)
fig, axes = plt.subplots(1, 2, figsize=(20, 13))
t = jnp.logspace(-3, 0, 1000, endpoint=False)
for i, alpha in enumerate(jnp.linspace(0.01, 1, 10)):
    for j, beta in enumerate(tqdm(jnp.linspace(0, 1, 10))):
        globals()["a{}{}".format(i + 1, j + 1)] = [MLF1(alpha, beta, -z) for z in t]
        print(globals()["a{}{}".format(i + 1, j + 1)][0])
        axes[0].plot(t, globals()["a{}{}".format(i + 1, j + 1)], label=f"{beta}")
        axes[1].plot(t, ml(-t, alpha=alpha, beta=beta), label=f"{beta}")

# axes[0].legend()
# axes[1].legend()
# %%
jnp.angle(1)
# %%
### (0<alpha<=1) & (jnp.abs(z)>jnp.floor(10+5*alpha))
fig, axes = plt.subplots(1, 2, figsize=(20, 13))
t = jnp.arange(16, 70, 1e-1)
for i, alpha in enumerate(jnp.linspace(0.01, 1, 10)):
    for j, beta in enumerate(tqdm(jnp.linspace(0, 1, 10))):
        globals()["b{}{}".format(i + 1, j + 1)] = [MLF2(alpha, beta, -z) for z in t]
        print(globals()["b{}{}".format(i + 1, j + 1)][0])
        # print(ml(-t, alpha=alpha, beta=beta))
        axes[0].plot(t, globals()["b{}{}".format(i + 1, j + 1)], label=f"{alpha}")
        axes[1].plot(t, ml(-t, alpha=alpha, beta=beta), label=f"{alpha}")
# axes[0].legend()
# axes[1].legend()
# %%
### (0<alpha<=1) & (1<=jnp.abs(z)<=jnp.floor(10+5*alpha)) & (beta<=1)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
t = jnp.arange(1, 10, 1e-1)
for i, alpha in enumerate(jnp.linspace(0.1, 1, 10)):
    for j, beta in enumerate(jnp.linspace(0, 1, 10)):
        globals()["c{}{}".format(i + 1, j + 1)] = [MLF3(alpha, beta, -z) for z in t]
        print(globals()["c{}{}".format(i + 1, j + 1)])
        ax.plot(t, globals()["c{}{}".format(i + 1, j + 1)])
# %%
###
a11
for i, a in enumerate(alpha):
    globals()["a{}".format(i + 1)] = [MLF(a, beta, -i) for i in t]
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, a1, label="alpha = 0.25")
ax.plot(t, a2, label="alpha = 0.5")
ax.plot(t, a3, label="alpha = 0.75")
ax.plot(t, a4, label="alpha = 1.0")
ax.legend()
# %%
len(jnp.linspace(1, 10, 10))
# %%
for i, alpha in enumerate(jnp.linspace(0.01, 1, 10)):
    for j, beta in enumerate(jnp.linspace(-10, 10, 10)):
        globals()["b{}{}".format(i + 1, j + 1)] = [1]
        print(i, j)
# %%
b11
# %%
