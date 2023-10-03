# %%
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from neuralconstitutive.tipgeometry import Conical
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


class HonerkampWeeseBimodal(eqx.Module):
    A: float
    B: float
    t_x: float
    t_y: float
    t_a: float
    t_b: float
    n: int

    def __init__(
        self,
        A: float = 0.1994711402,
        B: float = 0.1994711402,
        t_x: float = 5e-2,
        t_y: float = 5.0,
        t_a: float = 1e-3,
        t_b: float = 1e2,
        n: int = 100,
    ):
        # Default values for A, B correspond to 1/(2*sqrt(2*pi))
        self.A = A
        self.B = B
        self.t_x = t_x
        self.t_y = t_y
        self.t_a = t_a
        self.t_b = t_b
        self.n = n

    def __call__(self):
        t_i = jnp.logspace(jnp.log10(self.t_a), jnp.log10(self.t_b), self.n)
        h_i = self.A * jnp.exp(-0.5 * jnp.log(t_i / self.t_x) ** 2) + self.B * jnp.exp(
            -0.5 * jnp.log(t_i / self.t_y) ** 2
        )
        return t_i, h_i
# %%
spectrum = HonerkampWeeseBimodal()
t_i, h_i = spectrum()
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, ".")
ax.set_xscale("log", base=10)
# %%
#@partial(jax.vmap, in_axes=(0, None, None))
def function_from_discrete_spectra(t: float, t_i: jax.Array, h_i: jax.Array):
    h0 = jnp.log(t_i[1]) - jnp.log(t_i[0])
    return jnp.dot(h_i * h0, jnp.exp(-t / t_i))
# %%
t = jnp.arange(0.0, 10.0, 0.001)
#g = function_from_discrete_spectra(t, *spectrum())
phi = partial(function_from_discrete_spectra, t_i=spectrum()[0],h_i=spectrum()[1])
# phi = jax.vmap(phi)(t)
# %%
# plt.plot(t, phi, ".") 
# %%
def DLNM_Integrand(t_, t, phi, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return(
        a
        *b
        * phi(t-t_)
        * velocity(t_)
        * indentation(t_) ** (b - 1)
    )

def F_app_integral(t__, phi_, velocity_, indentation_, tip_):
    F = []
    for i in tqdm(t__):
        integrand_ = partial(
            DLNM_Integrand,
            t=i,
            phi=phi_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F    

def F_ret_integral(t__, t___, phi_, velocity_, indentation_, tip_):
    F = []
    for i, j in zip(tqdm(t__), t___):
        integrand_ = partial(
            DLNM_Integrand,
            t=j,
            phi=phi_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F
#%%
t = jnp.arange(0.0, 10.0, 0.001)
theta = (18.0 / 180.0) * np.pi
tip = Conical(theta)
# %%
def v_app(t_):
    v = 10 * 1e-6
    return v


def v_ret(t_):
    v = 10 * 1e-6
    return -v


def i_app(t_):
    return v_app(t_) * t_


def i_ret(t_):
    return (v_app(t_) * 0.2) - v_ret(t_) * (t_ - 0.2)
# %%
# F = F_app_integral(t, phi, v_app, i_app, tip)
# %%
# plt.plot(t, F)
#%%
space = 101  # odd number
idx = int((space - 1) / 2)
t_array = np.linspace(0, 10, space)
t_app = t_array[: idx + 1]
t_ret = t_array[idx:]
t_max = t_array[idx]

# Find t1
def Integrand(t_, t, phi, velocity):
    return phi(t-t_) * velocity(t_)

def Quadrature(t1, t_, t_max_, phi_, velocity_app, velocity_ret):
    integrand_app = partial(
        Integrand, t=t_, phi=phi_, velocity=velocity_app
    )
    integrand_ret = partial(
        Integrand, t=t_, phi=phi_, velocity=velocity_ret
    )
    quad_app = quad(integrand_app, t1, t_max_)
    quad_ret = quad(integrand_ret, t_max_, t_)
    return quad_app[0] + quad_ret[0]

def Calculation_t1(
    time_ret, t_max__, phi__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in tqdm(time_ret):
        quadrature = partial(
            Quadrature,
            t_=i,
            t_max_=t_max__,
            phi_=phi__,
            velocity_app=velocity_app__,
            velocity_ret=velocity_ret__,
        )
        if quadrature(t1=0) * quadrature(t1=t_max__) > 0:
            t1.append(0)
        else:
            t_1 = root_scalar(quadrature, method="bisect", bracket=(0, t_max))
            t1.append(t_1.root)
    return t1

t_1 = Calculation_t1(
    time_ret=t_ret,
    t_max__=t_max,
    phi__=phi,
    velocity_app__=v_app,
    velocity_ret__=v_ret,)
#%%
F_app = F_app_integral(
    t__=t_app,
    phi_=phi,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
#%%
F_ret = F_ret_integral(
    t__=t_1,
    t___=t_ret,
    phi_=phi,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)    
#%%
F_app = np.array(F_app)
F_ret = np.array(F_ret)
F_app[np.isnan(F_app)] = 0
F_ret[np.isnan(F_ret)] = 0
F_total = np.append(F_app, F_ret[1:])

plt.plot(t_array, F_total)
# %%