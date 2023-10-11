#%%
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import gamma
from scipy.optimize import curve_fit
import scipy.special as sc
from neuralconstitutive.tipgeometry import Conical
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
#%%
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
ax.set_xlabel("Relaxation Time τ[s]")
ax.set_ylabel("Relaxation Spectrum H(τ)[Pa]")
# %%
#@partial(jax.vmap, in_axes=(0, None, None))
def function_from_discrete_spectra(t: float, t_i: jax.Array, h_i: jax.Array):
    h0 = jnp.log(t_i[1]) - jnp.log(t_i[0])
    return jnp.dot(h_i * h0, jnp.exp(-t / t_i))
# %%
t = jnp.arange(1e-2, 100.0, 0.0001)
#g = function_from_discrete_spectra(t, *spectrum())
phi = partial(function_from_discrete_spectra, t_i=spectrum()[0],h_i=spectrum()[1])
g = jax.vmap(phi)(t)
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t, g)
ax.set_xlabel("Time[s]")
ax.set_ylabel("Relaxation Function[Pa]")
# %%
# Relaxation function curve fitting
def PLR(t, E0,  alpha, t_prime):
    return E0 * (1 + t / t_prime) ** (-alpha)


def SLS(t, E0, E_inf, tau):
    return E_inf + (E0 - E_inf) * np.exp(-(t / tau))


def KWW(t, E0, E_inf, tau, beta):
    return E_inf + (E0 - E_inf) * np.exp(-((t / tau) ** beta))

def Fung(t, E0, C, tau1, tau2):
    return E0 * (
        (1 + C * (sc.exp1(t / tau2) - sc.exp1(t / tau1)))
        / (1 + C * np.log(tau2 / tau1))
    )
plr = partial(PLR, t_prime=1e-5)
popt_plr, pcov_plr = curve_fit(plr, t, g)
popt_sls, pcov_sls = curve_fit(SLS, t, g)
popt_kww, pcov_kww = curve_fit(KWW, t, g, maxfev=1000000000)
popt_fung, pcov_fung = curve_fit(Fung, t, g)
#%%
# Relaxation Function Graph(RS curve fit)
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t, PLR(t, 1e-5, *popt_plr), label="PLR")
ax.plot(t, SLS(t, *popt_sls), label="SLS")
ax.plot(t, KWW(t, *popt_kww), label="KWW")
ax.plot(t, Fung(t, *popt_fung), label="Fung")
ax.plot(t, g, label="DLNM")
# ax.set_xscale("log")
ax.legend()
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
    return v_ret(t_-2*t_[0])
# %%
# F = F_app_integral(t, phi, v_app, i_app, tip)
# %%
# plt.plot(t, F)
#%%
# Tip Geometry
theta = (18.0 / 180.0) * np.pi
tip = Conical(theta)
#%%
# Data points
space = 201  # odd number
idx = int((space - 1) / 2)
t_array = np.linspace(1e-5, 10, space)
t_app = t_array[: idx + 1]
t_ret = t_array[idx:]
t_max = t_array[idx]
#%%
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
F_total = F_total * 1e9
#%%
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t_array, F_total)

#%%
def PLR_constit_integand(t_, t, E0, alpha, t_prime, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return (
        a
        * b
        * E0
        * (1 + (t - t_) / t_prime) ** (-alpha)
        * velocity(t_)
        * indentation(t_) ** (b - 1)
    )

def SLS_constit_integand(t_, t, E0, E_inf, tau, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return (
        a
        * b
        * (E_inf + (E0 - E_inf) * np.exp(-(t - t_) / tau))
        * velocity(t_)
        * indentation(t_) ** (b - 1)
    )

def Fung_constit_integrand(t_, t, E0, C, tau1, tau2, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return (
        a
        * b
        * E0
        * (
            (1 + C * (sc.exp1((t - t_) / tau2) - sc.exp1((t - t_) / tau1)))
            / (1 + C * np.log(tau2 / tau1))
        )
        * velocity(t_)
        * indentation(t_) ** (b - 1)
    )

def Integrand_PLR(t_, t, E0, t_prime, alpha, velocity):
    return (E0 * (1 + (t - t_) / t_prime) ** (-alpha)) * velocity(t_)

def Integrand_SLS(t_, t, E0, E_inf, tau, velocity):
    return (E_inf + (E0 - E_inf) * np.exp(-(t - t_) / tau)) * velocity(t_)

def Integrand_Fung(t_, t, E0, C, tau1, tau2, velocity):
    return (
        E0
        * (
            (1 + C * (sc.exp1((t - t_) / tau2) - sc.exp1((t - t_) / tau1)))
            / (1 + C * np.log(tau2 / tau1))
        )
    ) * velocity(t_)
#%%
def Quadrature_PLR(t1, t_, t_max_, E0_, t_prime_, alpha_, velocity_app, velocity_ret):
    integrand_app = partial(
        Integrand_PLR, E0=E0_, t=t_, t_prime=t_prime_, alpha=alpha_, velocity=velocity_app
    )
    integrand_ret = partial(
        Integrand_PLR, E0=E0_, t=t_, t_prime=t_prime_, alpha=alpha_, velocity=velocity_ret
    )
    quad_app = quad(integrand_app, t1, t_max_)
    quad_ret = quad(integrand_ret, t_max_, t_)
    return quad_app[0] + quad_ret[0]

def Quadrature_SLS(t1, t_, t_max_, E0_, E_inf_, tau_, velocity_app, velocity_ret):
    integrand_app = partial(
        Integrand_SLS, E0=E0_, t=t_, E_inf=E_inf_, tau=tau_, velocity=velocity_app
    )
    integrand_ret = partial(
        Integrand_SLS, E0=E0_, t=t_, E_inf=E_inf_, tau=tau_, velocity=velocity_ret
    )
    quad_app = quad(integrand_app, t1, t_max_)
    quad_ret = quad(integrand_ret, t_max_, t_)
    return quad_app[0] + quad_ret[0]

def Quadrature_Fung(t1, t_, t_max_, E0_, C_, tau1_, tau2_, velocity_app, velocity_ret):
    integrand_app = partial(
        Integrand_Fung,
        E0=E0_,
        t=t_,
        C=C_,
        tau1=tau1_,
        tau2=tau2_,
        velocity=velocity_app,
    )
    integrand_ret = partial(
        Integrand_Fung,
        E0=E0_,
        t=t_,
        C=C_,
        tau1=tau1_,
        tau2=tau2_,
        velocity=velocity_ret,
    )
    quad_app = quad(integrand_app, t1, t_max_)
    quad_ret = quad(integrand_ret, t_max_, t_)
    return quad_app[0] + quad_ret[0]

def Calculation_t1_PLR(
    time_ret, t_max__, E0__, t_prime__, alpha__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in time_ret:
        quadrature = partial(
            Quadrature_PLR,
            t_=i,
            t_max_=t_max__,
            E0_=E0__,
            t_prime_=t_prime__,
            alpha_=alpha__,
            velocity_app=velocity_app__,
            velocity_ret=velocity_ret__,
        )
        if quadrature(t1=0) * quadrature(t1=t_max__) > 0:
            t1.append(0)
        else:
            t_1 = root_scalar(quadrature, method="bisect", bracket=(0, t_max))
            t1.append(t_1.root)
    return t1

def Calculation_t1_SLS(
    time_ret, t_max__, E0__, E_inf__, tau__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in time_ret:
        quadrature = partial(
            Quadrature_SLS,
            t_=i,
            t_max_=t_max__,
            E0_=E0__,
            E_inf_=E_inf__,
            tau_=tau__,
            velocity_app=velocity_app__,
            velocity_ret=velocity_ret__,
        )
        if quadrature(t1=0) * quadrature(t1=t_max__) > 0:
            t1.append(0)
        else:
            t_1 = root_scalar(quadrature, method="bisect", bracket=(0, t_max))
            t1.append(t_1.root)
    return t1

def Calculation_t1_Fung(
    time_ret, t_max__, E0__, C__, tau1__, tau2__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in time_ret:
        quadrature = partial(
            Quadrature_Fung,
            t_=i,
            t_max_=t_max__,
            E0_=E0__,
            C_=C__,
            tau1_=tau1__,
            tau2_=tau2__,
            velocity_app=velocity_app__,
            velocity_ret=velocity_ret__,
        )
        if quadrature(t1=0) * quadrature(t1=t_max__) > 0:
            t1.append(0)
        else:
            t_1 = root_scalar(quadrature, method="bisect", bracket=(0, t_max))
            t1.append(t_1.root)
    return t1
#%%
# Total Force
def F_total_integral_PLR(
    time, E0_, alpha_, max_ind, t_prime_, indentation_, tip_, velocity_app, velocity_ret
):
    time_app = time[: max_ind + 1]
    time_ret = time[max_ind:]
    t1 = Calculation_t1_PLR(
        time_ret, time_ret[0], E0_, t_prime_, alpha_, velocity_app, velocity_ret
    )
    F_app = []
    for i in time_app:
        integrand_ = partial(
            PLR_constit_integand,
            t=i,
            E0=E0_,
            alpha=alpha_,
            t_prime=t_prime_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_app.append(quad(integrand_, 0, i)[0])

    F_ret = []
    for i, j in zip(time_ret, t1):
        integrand_ = partial(
            PLR_constit_integand,
            t=i,
            E0=E0_,
            alpha=alpha_,
            t_prime=t_prime_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_ret.append(quad(integrand_, 0, j)[0])

    F = F_app + F_ret[1:]
    F = np.array(F) 
    return F * 1e9

def F_total_integral_SLS(
    time, E0_, E_inf_, tau_,
      max_ind, indentation_, tip_, velocity_app, velocity_ret
):
    time_app = time[: max_ind + 1]
    time_ret = time[max_ind:]
    t1 = Calculation_t1_SLS(
        time_ret, time_ret[0], E0_, E_inf_, tau_, velocity_app, velocity_ret
    )
    F_app = []
    for i in time_app:
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_app.append(quad(integrand_, 0, i)[0])

    F_ret = []
    for i, j in zip(time_ret, t1):
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_ret.append(quad(integrand_, 0, j)[0])

    F = F_app + F_ret[1:]
    F = np.array(F)
    return F * 1e9

def F_total_integral_Fung(
    time, C_, E0_, tau1_, tau2_, max_ind, indentation_, tip_, velocity_app, velocity_ret
):
    time_app = time[: max_ind + 1]
    time_ret = time[max_ind:]
    t1 = Calculation_t1_Fung(
        time_ret, time_ret[0], E0_, C_, tau1_, tau2_, velocity_app, velocity_ret
    )
    F_app = []
    for i in time_app:
        integrand_ = partial(
            Fung_constit_integrand,
            t=i,
            E0=E0_,
            C=C_,
            tau1=tau1_,
            tau2=tau2_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_app.append(quad(integrand_, 0, i)[0])

    F_ret = []
    for i, j in zip(time_ret, t1):
        integrand_ = partial(
            Fung_constit_integrand,
            t=i,
            E0=E0_,
            C=C_,
            tau1=tau1_,
            tau2=tau2_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_ret.append(quad(integrand_, 0, j)[0])

    F = F_app + F_ret[1:]
    F = np.array(F)
    return F * 1e9
#%%
F_total_func_PLR = partial(
    F_total_integral_PLR,
    max_ind=idx,
    t_prime_=1e-5,
    indentation_=i_app,
    velocity_app=v_app,
    velocity_ret=v_ret,
    tip_=tip,
)

F_total_func_Fung = partial(
    F_total_integral_Fung,
    max_ind=idx,
    indentation_=i_app,
    velocity_app=v_app,
    velocity_ret=v_ret,
    tip_=tip,
)

F_total_func_SLS = partial(
    F_total_integral_SLS,
    max_ind=idx,
    indentation_=i_app,
    velocity_app=v_app,
    velocity_ret=v_ret,
    tip_=tip,
)

# %%
popt_total_plr, pcov_total_plr = curve_fit(F_total_func_PLR, t_array, F_total)
popt_total_fung, pcov_total_fung = curve_fit(F_total_func_Fung, t_array, F_total)
popt_total_sls, pcov_total_sls = curve_fit(F_total_func_SLS, t_array, F_total)

#%%
F_total_curvefit_plr = F_total_func_PLR(t_array, *popt_total_plr)
F_total_curvefit_fung = F_total_func_Fung(t_array, *popt_total_fung)
F_total_curvefit_sls = F_total_func_SLS(t_array, *popt_total_sls)
#%%
print(popt_total_fung, popt_total_sls, popt_total_plr)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t_array, F_total, label="DLNM")
ax.plot(t_array, F_total_curvefit_plr, label = "PLR")
ax.plot(t_array, F_total_curvefit_sls, label="SLS")
ax.plot(t_array, F_total_curvefit_fung, label="Fung")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Force[nN]")
ax.legend()
#%%
def RS_PLR(tau, t_prime, E0, alpha):
    return E0*t_prime*(np.exp(-t_prime/tau))*(t_prime/tau)**(alpha-1)/tau/gamma(alpha)

def RS_Fung(tau, C, tau1, tau2):
    a = np.zeros(len(tau))
    idx1 = np.where(tau1<tau)[0][0]
    idx2 = np.where(tau<tau2)[0][-1]
    a[idx1:idx2+1] = C
    return a

spectrum_PLR = RS_PLR(t_i, 1e-5, *popt_total_plr)
spectrum_Fung = RS_Fung(t_i, popt_total_fung[0], popt_total_fung[2], popt_total_fung[3])
#%%
# Relaxation Function Graph

#%%
# Relaxation Time Spectrum Graph
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, label="DLNM")
ax.plot(t_i, spectrum_PLR, label="PLR")
ax.axvline(popt_total_sls[2],label="SLS")
ax.plot(t_i, spectrum_Fung/1000, label="Fung")
ax.set_xscale("log", base=10)
ax.set_xlabel("Relaxation Time τ[s]")
ax.set_ylabel("Relaxation Spectrum H(τ)[Pa]")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot()
#%%
popt_total_plr, popt_plr
# %%
popt_total_sls, popt_sls
#%%
popt_total_fung, popt_fung
# %%
