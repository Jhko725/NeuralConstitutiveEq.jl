# %%
# ruff: noqa: F722
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import scipy.special as sc
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx
import optimistix as optx

from neuralconstitutive.tipgeometry import AbstractTipGeometry, Conical
from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
    Fung,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.ting import force_ting
from neuralconstitutive.plotting import plot_relaxation_fn, plot_indentation

jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
def fit_relaxation_fn(
    constitutive: AbstractConstitutiveEqn,
    time: Float[Array, " len_data"],
    relaxation: Float[Array, " len_data"],
    solver,
    *,
    max_steps: int = 1024
):

    def residual(constitutive: AbstractConstitutiveEqn, args):
        (time,) = args
        del args
        relaxation_pred = eqx.filter_vmap(constitutive.relaxation_function)(time)
        return relaxation_pred - relaxation

    args = (time,)
    result = optx.least_squares(
        residual, solver, constitutive, args, max_steps=max_steps
    )
    return result


@eqx.filter_jit
def fit_force(
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
    indentations: tuple[Indentation, Indentation],
    forces: tuple[Float[Array, " len_app"], Float[Array, " len_ret"]],
    solver,
    *,
    max_steps=1024
):
    def residual(constitutive, args):
        tip, indentations, forces = args
        del args
        f_pred = jnp.concatenate(forces)
        f_true = force_ting(constitutive, tip, *indentations)
        f_true = jnp.concatenate(f_true)
        return f_pred - f_true

    args = (tip, indentations, forces)
    result = optx.least_squares(
        residual, solver, constitutive, args, max_steps=max_steps
    )
    return result


# %%
def PLR(t, E0, t_prime, alpha):
    return E0 * (1 + t / t_prime) ** (-alpha)


def SLS(t, E0, E_inf, tau):
    return E_inf + (E0 - E_inf) * np.exp(-(t / tau))


def KWW(t, E0, E_inf, tau, beta):
    return E_inf + (E0 - E_inf) * np.exp(-((t / tau) ** beta))


def Fung2(t, E0, C, tau1, tau2):
    return E0 * (
        (1 + C * (sc.exp1(t / tau2) - sc.exp1(t / tau1)))
        / (1 + C * np.log(tau2 / tau1))
    )


# %%
t = np.linspace(1e-2, 1e2, 1000)
# %%
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = plot_relaxation_fn(ax, plr, t)
# %%
plr_data = eqx.filter_vmap(plr.relaxation_function)(t)
# SLS model fitting(3 version)
sls = StandardLinearSolid(100.0, 1.0, 1.0)
kww = KohlrauschWilliamsWatts(100.0, 1.0, 10.0, 1.0)
fung = Fung(10.0, 0.01, 1.0, 1.0)
solver = optx.LevenbergMarquardt(atol=1e-3, rtol=1e-3)
# %%
result = fit_relaxation_fn(sls, t, plr_data, solver)
# %%
result = fit_relaxation_fn(kww, t, plr_data, solver)
# %%
result = fit_relaxation_fn(fung, t, plr_data, solver)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = plot_relaxation_fn(ax, result.value, t)
ax.plot(t, plr_data, ".", label="Ground truth")
ax.legend()
# %%
## Figure out how to do curve fit with latin hypercube sampling

# E0_sls1, E0_sls2, E0_sls3 = 60, 70, 80
# E_inf_sls1, E_inf_sls2, E_inf_sls3 = 5, 25, 40
tau_sls1, tau_sls2, tau_sls3 = 15.0, 20.0, 30.0
# %%
# tau_sls1, pcov_sls1 = curve_fit(lambda t, tau : SLS(t, E0_sls1, E_inf_sls, tau), t, sls_t)
# tau_sls2, pcov_sls2 = curve_fit(lambda t, tau : SLS(t, E0_sls2, E_inf_sls, tau), t, sls_t)
# tau_sls3, pcov_sls3 = curve_fit(lambda t, tau : SLS(t, E0_sls3, E_inf_sls, tau), t, sls_t)

# sls_t_1 = SLS(t, E0_sls1, popt_sls[1], popt_sls1)
# sls_t_2 = SLS(t, E0_sls2, popt_sls[1], popt_sls2)
# sls_t_3 = SLS(t, E0_sls3, popt_sls[1], popt_sls3)

# E0_sls1, pcov_sls1 = curve_fit(lambda t, E0 : SLS(t, E0_sls, E_inf_sls1, tau_sls), t, sls_t)
# E0_sls2, pcov_sls2 = curve_fit(lambda t, E0 : SLS(t, E0_sls, E_inf_sls2, tau_sls), t, sls_t)
# E0_sls3, pcov_sls3 = curve_fit(lambda t, E0 : SLS(t, E0_sls, E_inf_sls3, tau_sls), t, sls_t)

# sls_t_1 = SLS(t, E0_sls1, E_inf_sls1, tau_sls)
# sls_t_2 = SLS(t, E0_sls2, E_inf_sls2, tau_sls)
# sls_t_3 = SLS(t, E0_sls3, E_inf_sls3, tau_sls)

E_inf_sls1, pcov_sls1 = curve_fit(
    lambda t, E_inf: SLS(t, E0_sls, E_inf, tau_sls1), t, sls_t
)
E_inf_sls2, pcov_sls2 = curve_fit(
    lambda t, E_inf: SLS(t, E0_sls, E_inf, tau_sls2), t, sls_t
)
E_inf_sls3, pcov_sls3 = curve_fit(
    lambda t, E_inf: SLS(t, E0_sls, E_inf, tau_sls3), t, sls_t
)

sls_t_1 = SLS(t, E0_sls, E_inf_sls1, tau_sls1)
sls_t_2 = SLS(t, E0_sls, E_inf_sls2, tau_sls2)
sls_t_3 = SLS(t, E0_sls, E_inf_sls3, tau_sls3)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, sls_t)
ax.plot(t, sls_t_1)
ax.plot(t, sls_t_2)
ax.plot(t, sls_t_3)
ax.plot(t, plr_t)
ax.set_xscale("log")
# %%
popt_kww, pcov_kww = curve_fit(
    lambda t, E_inf, tau, beta: KWW(t, E0, E_inf, tau, beta),
    t,
    plr_t,
    maxfev=1000000,
    p0=[1, 1, 1],
)
popt_kww
# %%
kww_t = KWW(t, E0, *popt_kww)
# %%
popt_fung, pcov_fung = curve_fit(Fung, t, plr_t)
popt_fung
# %%
fung_t = Fung(t, *popt_fung)

plot_kwargs = {"linewidth": 1.0, "alpha": 0.8, "markersize": 3.0}
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.plot(t, sls_t)
ax.plot(t, plr_t / 1000, ".-", label="PLR", **plot_kwargs)
ax.plot(t, kww_t / 1000, ".-", label="KWW", **plot_kwargs)
ax.plot(t, sls_t / 1000, ".-", label="SLS", **plot_kwargs)
ax.plot(t, fung_t / 1000, ".-", label="Fung", **plot_kwargs)
ax.set_xlabel("Time[s]")
ax.set_ylabel("Stress Relaxation Function[kPa]")
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid(color="lightgray", linestyle="--")
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.legend()


# %%
# Force reconstruction
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


def KWW_constit_integrand(t_, t, E0, E_inf, tau, beta, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return (
        a
        * b
        * (E_inf + (E0 - E_inf) * np.exp(-(((t - t_) / tau) ** beta)))
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


def F_app_integral(t__, E0_, alpha_, t_prime_, velocity_, indentation_, tip_):
    F = []
    for i in t__:
        integrand_ = partial(
            PLR_constit_integand,
            t=i,
            E0=E0_,
            alpha=alpha_,
            t_prime=t_prime_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_app_integral_SLS(t__, E0_, E_inf_, tau_, velocity_, indentation_, tip_):
    F = []
    for i in t__:
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_app_integral_KWW(t__, E0_, E_inf_, tau_, beta_, velocity_, indentation_, tip_):
    F = []
    for i in t__:
        integrand_ = partial(
            KWW_constit_integrand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            beta=beta_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_app_integral_Fung(t__, E0_, C_, tau1_, tau2_, velocity_, indentation_, tip_):
    F = []
    for i in t__:
        integrand_ = partial(
            Fung_constit_integrand,
            t=i,
            E0=E0_,
            C=C_,
            tau1=tau1_,
            tau2=tau2_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_ret_integral(t__, t___, E0_, alpha_, t_prime_, velocity_, indentation_, tip_):
    F = []
    for i, j in zip(t__, t___):
        integrand_ = partial(
            PLR_constit_integand,
            t=j,
            E0=E0_,
            alpha=alpha_,
            t_prime=t_prime_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_ret_integral_SLS(t__, t___, E0_, E_inf_, tau_, velocity_, indentation_, tip_):
    F = []
    for i, j in zip(t__, t___):
        integrand_ = partial(
            SLS_constit_integand,
            t=j,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_ret_integral_KWW(
    t__, t___, E0_, E_inf_, tau_, beta_, velocity_, indentation_, tip_
):
    F = []
    for i, j in zip(t__, t___):
        integrand_ = partial(
            KWW_constit_integrand,
            t=j,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            beta=beta_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_ret_integral_Fung(
    t__, t___, E0_, C_, tau1_, tau2_, velocity_, indentation_, tip_
):
    F = []
    for i, j in zip(t__, t___):
        integrand_ = partial(
            Fung_constit_integrand,
            t=j,
            E0=E0_,
            C=C_,
            tau1=tau1_,
            tau2=tau2_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


# %%
# Find t1
def Integrand(t_, t, E0, t_prime, alpha, velocity):
    return (E0 * (1 + (t - t_) / t_prime) ** (-alpha)) * velocity(t_)


def Integrand_SLS(t_, t, E0, E_inf, tau, velocity):
    return (E_inf + (E0 - E_inf) * np.exp(-(t - t_) / tau)) * velocity(t_)


def Integrand_KWW(t_, t, E0, E_inf, tau, beta, velocity):
    return (E_inf + (E0 - E_inf) * np.exp(-(((t - t_) / tau) ** beta))) * velocity(t_)


def Integrand_Fung(t_, t, E0, C, tau1, tau2, velocity):
    return (
        E0
        * (
            (1 + C * (sc.exp1((t - t_) / tau2) - sc.exp1((t - t_) / tau1)))
            / (1 + C * np.log(tau2 / tau1))
        )
    ) * velocity(t_)


# %%
def Quadrature(t1, t_, t_max_, E0_, t_prime_, alpha_, velocity_app, velocity_ret):
    integrand_app = partial(
        Integrand, E0=E0_, t=t_, t_prime=t_prime_, alpha=alpha_, velocity=velocity_app
    )
    integrand_ret = partial(
        Integrand, E0=E0_, t=t_, t_prime=t_prime_, alpha=alpha_, velocity=velocity_ret
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


def Quadrature_KWW(
    t1, t_, t_max_, E0_, E_inf_, tau_, beta_, velocity_app, velocity_ret
):
    integrand_app = partial(
        Integrand_KWW,
        E0=E0_,
        t=t_,
        E_inf=E_inf_,
        tau=tau_,
        beta=beta_,
        velocity=velocity_app,
    )
    integrand_ret = partial(
        Integrand_KWW,
        E0=E0_,
        t=t_,
        E_inf=E_inf_,
        tau=tau_,
        beta=beta_,
        velocity=velocity_ret,
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


# %%
def Calculation_t1(
    time_ret, t_max__, E0__, t_prime__, alpha__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in time_ret:
        quadrature = partial(
            Quadrature,
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


def Calculation_t1_KWW(
    time_ret, t_max__, E0__, E_inf__, tau__, beta__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in time_ret:
        quadrature = partial(
            Quadrature_KWW,
            t_=i,
            t_max_=t_max__,
            E0_=E0__,
            E_inf_=E_inf__,
            tau_=tau__,
            beta_=beta__,
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


# %%
# Make indentation (with normalized time and indentation)
dt = 1e-2
t_app = jnp.arange(0.0, 1.0 + dt, dt)
t_ret = jnp.arange(1.0, 2.0 + dt, dt)
h_app = 1.0 * t_app
h_ret = 1.0 * (2.0 - t_ret)
app, ret = Indentation(t_app, h_app), Indentation(t_ret, h_ret)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = plot_indentation(ax, app, marker=".")
ax = plot_indentation(ax, ret, marker=".")
ax
# %%
tip = Conical(jnp.pi / 10)  # in radians
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)

force_plr = force_ting(plr, tip, app, ret)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, force_plr[0], ".")
ax.plot(ret.time, force_plr[1], ".")
# %%
sls = StandardLinearSolid(100.0, 1.0, 1.0)
solver = optx.LevenbergMarquardt(atol=1e-3, rtol=1e-3)
result_sls = fit_force(sls, tip, (app, ret), force_plr, solver)


# %%
@eqx.filter_jit
def force_objective(constitutive, tip, indentations, forces):
    f_pred = jnp.concatenate(forces)
    f_true = force_ting(constitutive, tip, *indentations)
    f_true = jnp.concatenate(f_true)
    return jnp.sum((f_pred - f_true) ** 2)


val = eqx.filter_grad(force_objective)(sls, tip, (app, ret), force_plr)
print(val)
# %%
E_inf, tau, beta = popt_kww
E0_sls, E_inf_sls, tau_sls = popt_sls
E0_fung, C_fung, tau1_fung, tau2_fung = popt_fung
# %%
space = 1001  # odd number
idx = int((space - 1) / 2)
t_array = np.linspace(1e-2, 1e2, space)
t_app = t_array[: idx + 1]
t_ret = t_array[idx:]
t_max = t_array[idx]


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
t_1 = Calculation_t1(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E_0,
    t_prime__=t_prime,
    alpha__=gamma,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)
t_1_kww = Calculation_t1_KWW(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E0,
    E_inf__=E_inf,
    tau__=tau,
    beta__=beta,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)

t_1_sls = Calculation_t1_SLS(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E0_sls,
    E_inf__=E_inf_sls,
    tau__=tau_sls,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)
t_1_sls1 = Calculation_t1_SLS(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E0_sls,
    E_inf__=E_inf_sls1,
    tau__=tau_sls1,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)
t_1_sls2 = Calculation_t1_SLS(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E0_sls,
    E_inf__=E_inf_sls2,
    tau__=tau_sls2,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)
t_1_sls3 = Calculation_t1_SLS(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E0_sls,
    E_inf__=E_inf_sls3,
    tau__=tau_sls3,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)

t_1_fung = Calculation_t1_Fung(
    time_ret=t_ret,
    t_max__=t_max,
    E0__=E0_fung,
    C__=C_fung,
    tau1__=tau1_fung,
    tau2__=tau2_fung,
    velocity_app__=v_app,
    velocity_ret__=v_ret,
)
# %%
F_app = F_app_integral(
    t__=t_app,
    E0_=E_0,
    alpha_=gamma,
    t_prime_=t_prime,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_app_sls = F_app_integral_SLS(
    t__=t_app,
    E0_=E0_sls,
    E_inf_=E_inf_sls,
    tau_=tau_sls,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_app_sls1 = F_app_integral_SLS(
    t__=t_app,
    E0_=E0_sls,
    E_inf_=E_inf_sls1,
    tau_=tau_sls1,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_app_sls2 = F_app_integral_SLS(
    t__=t_app,
    E0_=E0_sls,
    E_inf_=E_inf_sls2,
    tau_=tau_sls2,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_app_sls3 = F_app_integral_SLS(
    t__=t_app,
    E0_=E0_sls,
    E_inf_=E_inf_sls3,
    tau_=tau_sls3,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)


F_app_kww = F_app_integral_KWW(
    t__=t_app,
    E0_=E0,
    E_inf_=E_inf,
    tau_=tau,
    beta_=beta,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_app_fung = F_app_integral_Fung(
    t__=t_app,
    E0_=E0_fung,
    C_=C_fung,
    tau1_=tau1_fung,
    tau2_=tau2_fung,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)

F_ret = F_ret_integral(
    t__=t_1,
    t___=t_ret,
    E0_=E_0,
    alpha_=gamma,
    t_prime_=t_prime,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_ret_sls = F_ret_integral_SLS(
    t__=t_1_sls,
    t___=t_ret,
    E0_=E0_sls,
    E_inf_=E_inf_sls,
    tau_=tau_sls,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_ret_sls1 = F_ret_integral_SLS(
    t__=t_1_sls1,
    t___=t_ret,
    E0_=E0_sls,
    E_inf_=E_inf_sls1,
    tau_=tau_sls1,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_ret_sls2 = F_ret_integral_SLS(
    t__=t_1_sls2,
    t___=t_ret,
    E0_=E0_sls,
    E_inf_=E_inf_sls2,
    tau_=tau_sls2,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_ret_sls3 = F_ret_integral_SLS(
    t__=t_1_sls3,
    t___=t_ret,
    E0_=E0_sls,
    E_inf_=E_inf_sls3,
    tau_=tau_sls3,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)

F_ret_kww = F_ret_integral_KWW(
    t__=t_1_kww,
    t___=t_ret,
    E0_=E0,
    E_inf_=E_inf,
    tau_=tau,
    beta_=beta,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
F_ret_fung = F_ret_integral_Fung(
    t__=t_1_fung,
    t___=t_ret,
    E0_=E0_fung,
    C_=C_fung,
    tau1_=tau1_fung,
    tau2_=tau2_fung,
    velocity_=v_app,
    indentation_=i_app,
    tip_=tip,
)
# %%
F_app = np.array(F_app)
F_ret = np.array(F_ret)
F_app[np.isnan(F_app)] = 0
F_ret[np.isnan(F_ret)] = 0
F_total = np.append(F_app, F_ret[1:])

F_app_sls = np.array(F_app_sls)
F_ret_sls = np.array(F_ret_sls)
F_app_sls[np.isnan(F_app_sls)] = 0
F_ret_sls[np.isnan(F_ret_sls)] = 0
F_total_sls = np.append(F_app_sls, F_ret_sls[1:])

F_app_sls1 = np.array(F_app_sls1)
F_ret_sls1 = np.array(F_ret_sls1)
F_app_sls1[np.isnan(F_app_sls1)] = 0
F_ret_sls1[np.isnan(F_ret_sls1)] = 0
F_total_sls1 = np.append(F_app_sls1, F_ret_sls1[1:])

F_app_sls2 = np.array(F_app_sls2)
F_ret_sls2 = np.array(F_ret_sls2)
F_app_sls2[np.isnan(F_app_sls2)] = 0
F_ret_sls2[np.isnan(F_ret_sls2)] = 0
F_total_sls2 = np.append(F_app_sls2, F_ret_sls2[1:])

F_app_sls3 = np.array(F_app_sls3)
F_ret_sls3 = np.array(F_ret_sls3)
F_app_sls3[np.isnan(F_app_sls3)] = 0
F_ret_sls3[np.isnan(F_ret_sls3)] = 0
F_total_sls3 = np.append(F_app_sls3, F_ret_sls3[1:])


F_app_kww = np.array(F_app_kww)
F_ret_kww = np.array(F_ret_kww)
F_app_kww[np.isnan(F_app_kww)] = 0
F_ret_kww[np.isnan(F_ret_kww)] = 0
F_total_kww = np.append(F_app_kww, F_ret_kww[1:])

F_app_fung = np.array(F_app_fung)
F_ret_fung = np.array(F_ret_fung)
F_app_fung[np.isnan(F_app_fung)] = 0
F_ret_fung[np.isnan(F_ret_fung)] = 0
F_total_fung = np.append(F_app_fung, F_ret_fung[1:])

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
plot_kwargs = {"markersize": 3.0, "alpha": 0.2}
ax.plot(t_array, F_total * 1e9, ".", label="PLR", **plot_kwargs)
ax.plot(t_array, F_total_kww * 1e9, ".", label="KWW", **plot_kwargs)
ax.plot(t_array, F_total_sls * 1e9, ".", label="SLS", **plot_kwargs)
# ax.plot(t_array, F_total_sls1 * 1e9, '.', label="SLS1", markersize=markersize)
# ax.plot(t_array, F_total_sls2 * 1e9, '.', label="SLS2", markersize=markersize)
# ax.plot(t_array, F_total_sls3 * 1e9, '.', label="SLS3", markersize=markersize)
ax.plot(t_array, F_total_fung * 1e9, ".", label="Fung", **plot_kwargs)
ax.set_xlabel("Time[s]")
ax.set_ylabel("Force[nN]")
ax.grid(color="lightgray", linestyle="--")
ax.legend()
# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
plot_kwargs0 = {"linewidth": 1.0, "alpha": 0.8, "markersize": 3.0}

# ax.plot(t, sls_t)
axes[0].plot(t, plr_t / 1000, ".-", label="PLR", **plot_kwargs0)
axes[0].plot(t, kww_t / 1000, ".-", label="KWW", **plot_kwargs0)
axes[0].plot(t, sls_t / 1000, ".-", label="SLS", **plot_kwargs0)
axes[0].plot(t, fung_t / 1000, ".-", label="Fung", **plot_kwargs0)
axes[0].set_xscale("log")
axes[0].set_ylabel("Relaxation Function [kPa]")

plot_kwargs1 = {"markersize": 3.0, "alpha": 0.2}
axes[1].plot(t_array, F_total * 1e6, ".", label="PLR", **plot_kwargs1)
axes[1].plot(t_array, F_total_kww * 1e6, ".", label="KWW", **plot_kwargs1)
axes[1].plot(t_array, F_total_sls * 1e6, ".", label="SLS", **plot_kwargs1)
axes[1].plot(t_array, F_total_fung * 1e6, ".", label="Fung", **plot_kwargs1)
axes[1].set_ylabel("Force [μN]")
for ax in axes:
    ax.set_xlabel("Time [s]")
    ax.grid(color="lightgray", linestyle="--")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
axes[0].legend(ncols=2)

# ax.set_yscale("log")

# ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

# %%
