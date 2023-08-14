# %%
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import scipy.special as sc

from neuralconstitutive.tipgeometry import Conical


# %%
def PLR(t, E0, t_prime, alpha):
    return E0 * (1 + t / t_prime) ** (-alpha)


def SLS(t, E_inf, E0, tau):
    return E_inf + (E0 - E_inf) * np.exp(-(t / tau))


def KWW(t, E_inf, E0, tau, beta):
    return E_inf + (E0 - E_inf) * np.exp(-((t / tau) ** beta))


def Fung(t, C, E0, tau1, tau2):
    return E0 * (
        (1 + C * (sc.exp1(t / tau2) - sc.exp1(t / tau1)))
        / (1 + C * np.log(tau2 / tau1))
    )


# %%
# %%
t = np.logspace(-2, 2, 100)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, PLR(t, 4, 5e-4, 0.2), label="PLR")
ax.plot(t, KWW(t, 0.5, 2, 5.0, 0.5), label="KWW")
ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)
ax.set_yticks([0.25, 0.5, 1.0, 2.5])
ax.set_yticklabels([0.25, 0.5, 1.0, 2.5])
# %%
t = np.linspace(1e-2, 1e2, 200)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, PLR(t, 4, 5e-4, 0.2), label="PLR")
ax.plot(t, KWW(t, 0.4, 2, 1, 0.5), label="KWW")
ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)
ax.set_yticks([0.25, 0.5, 1.0, 2.5])
ax.set_yticklabels([0.25, 0.5, 1.0, 2.5])

# %%
E0 = 4
t_prime = 5e-4
alpha = 0.2
t = np.logspace(-2, 2, 100)
# %%
popt_kww, pcov_kww = curve_fit(KWW, t, plr_t, maxfev=10000000, p0=(0.4, 2, 1, 0.5))
# %%
plr_t = PLR(t, E0, t_prime, alpha)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, plr_t)
# %%
popt_sls, pcov_sls = curve_fit(SLS, t, plr_t)
popt_sls
# %%
sls_t = SLS(t, *popt_sls)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, sls_t)
ax.plot(t, plr_t)
ax.set_xscale("log")
# %%
popt_kww, pcov_kww = curve_fit(KWW, t, plr_t, maxfev=10000000, p0=(0.4, 3, 0.1, 0.3))
popt_kww
# %%
kww_t = KWW(t, *popt_kww)
# %%
popt_fung, pcov_fung = curve_fit(Fung, t, plr_t, maxfev=100000)
# %%
fung_t = Fung(t, *popt_fung)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# ax.plot(t, sls_t)
ax.plot(t, plr_t, ".", label="PLR_consit")
ax.plot(t, kww_t, label="KWW_constit")
ax.plot(t, sls_t, label="SLS_constit")
ax.plot(t, fung_t, label="Fung_constit")
ax.legend()


# ax.set_xscale("log")
# ax.set_yscale("log")
# %%
# Force reconstruction
# %%
# PLR model fitting
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


# %%
# Find t1
def Integrand(t_, E0, t, t_prime, alpha, velocity):
    return (E0 * (1 + (t - t_) / t_prime) ** (-alpha)) * velocity(t_)


def Integrand_KWW(t_, t, E0, E_inf, tau, beta, velocity):
    return (E_inf + (E0 - E_inf) * np.exp(-(((t - t_) / tau) ** beta))) * velocity(t_)


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


# %%
E_0 = 573  # Pa
gamma = 0.2
t_prime = 1e-5  # s
theta = (18.0 / 180.0) * np.pi
tip = Conical(theta)

# %%
E_inf, E0, tau, beta = popt_kww
# %%
space = 101  # odd number
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
# %%
F_app = np.array(F_app)
F_ret = np.array(F_ret)
F_app[np.isnan(F_app)] = 0
F_ret[np.isnan(F_ret)] = 0
F_total = np.append(F_app, F_ret[1:])

F_app_kww = np.array(F_app_kww)
F_ret_kww = np.array(F_ret_kww)
F_app_kww[np.isnan(F_app)] = 0
F_ret_kww[np.isnan(F_ret)] = 0
F_total_kww = np.append(F_app_kww, F_ret_kww[1:])
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(t_array, F_total * 1e9, ".", label="PLR")
ax.plot(t_array, F_total_kww * 1e9, ".", label="KWW")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
ax.legend()
# %%
