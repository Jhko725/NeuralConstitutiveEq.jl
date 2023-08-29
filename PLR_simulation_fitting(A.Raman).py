#%%
from functools import partial
from numpy import random
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, curve_fit
from scipy.interpolate import interp1d
import scipy.special as sc
import matplotlib.pyplot as plt
from neuralconstitutive.preprocessing import estimate_derivative
from neuralconstitutive.tipgeometry import Conical
#%%
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

# Find t1
def Integrand(t_, E0, t, t_prime, alpha, velocity):
    return (E0 * (1 + (t - t_) / t_prime) ** (-alpha)) * velocity(t_)
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

#%%
E_0 = 572 # Pa
gamma = 0.42
t_prime = 1e-5 # s
theta = (18.0/180.0)*np.pi
tip = Conical(theta)
#%%
space = 201 # odd number
idx = int((space-1)/2)
t_array = np.linspace(0, 0.4, space)
t_app = t_array[:idx+1]
t_ret = t_array[idx:]
t_max = t_array[idx]
#%%
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
    
#%%
t_1 = Calculation_t1(time_ret=t_ret, t_max__=t_max, E0__=E_0, t_prime__=t_prime, alpha__=gamma, velocity_app__=v_app, velocity_ret__=v_ret)
#%%
F_app = F_app_integral(t__=t_app, E0_=E_0, alpha_=gamma, t_prime_=t_prime, velocity_=v_app, indentation_= i_app, tip_=tip)
F_ret = F_ret_integral(t__=t_1, t___=t_ret, E0_=E_0, alpha_=gamma, t_prime_=t_prime, velocity_=v_app, indentation_=i_app, tip_=tip)
#%%
F_app = np.array(F_app)
F_ret = np.array(F_ret)
F_app[np.isnan(F_app)] = 0
F_ret[np.isnan(F_ret)] = 0
F_total = np.append(F_app, F_ret[1:])
#%%
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# ax.plot(t_array, F_total * 1e9, '.')
# ax.set_xlabel("Time(s)")
# ax.set_ylabel("Force(nN)")
# ax.set_title("Original Force")
#%%
# Generate Gaussian Noise
noise_amp = 0.01
mu, sigma = 0, F_ret[0]*noise_amp # average = 0, sigma = 0.1%
noise = np.random.normal(mu, sigma, np.shape(F_total))
F_total_noise = F_total + noise
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_array, F_total * 1e9, label="PLR")
ax.plot(t_array, F_total_noise * 1e9, '.', label= "with noise")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Force[nN]")
# ax.set_title(f"Force + {noise_amp *100}% noise (# of data = {space})")
ax.grid(color="lightgray", linestyle='--')
ax.legend()
#%%
v = 10 * 1e-6
indentation_app = v * t_app
indentation_ret = indentation_app[idx] - v * (t_ret-t_app[idx])
indentation = np.append(indentation_app, indentation_ret[1:])
#%%
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# ax.plot(t_array,  indentation * 1e6, '.')
# ax.set_xlabel("Time(s)")
# ax.set_ylabel("Indentation(μm)")
# ax.set_title("Original Indentation")
#%%
mu, sigma = 0, indentation_ret[0]*noise_amp # average = 0, sigma = 0.1%
noise = np.random.normal(mu, sigma, np.shape(indentation))
indentation_noise = indentation + noise
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_array, indentation * 1e6, label="Indentation")
ax.plot(t_array, indentation_noise * 1e6, '.', label= "with noise")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Indentation[μm]")
# ax.set_title(f"Indentation + {noise_amp *100}% noise (# of data = {space})")
ax.grid(color="lightgray", linestyle='--')
ax.legend()
#%%
# PLR_Curvefitting
force = F_total
indentation = indentation_noise
time = t_array
#%%
max_ind = np.argmax(indentation)
t_max = time[max_ind]
indent_max = indentation[max_ind]
# %%
F_app = force[:max_ind + 1]
F_ret = force[max_ind:]
# %%
# t_max 부분을 겹치게 해야 문제가 안생김
indentation_app = indentation[:max_ind + 1]
indentation_ret = indentation[max_ind:]

time_app = time[:max_ind + 1]
time_ret = time[max_ind:]

velocity_app = estimate_derivative(time_app, indentation_app)
velocity_ret = estimate_derivative(time_ret, indentation_ret)

indentation_app_func = interp1d(time_app, indentation_app)
indentation_ret_func = interp1d(time_ret, indentation_ret)
velocity_app_func = interp1d(time_app, velocity_app)
velocity_ret_func = interp1d(time_ret, velocity_ret)
# %%
# Truncation negrative force region
# negative_idx = np.where(F_ret < 0)[0]
# negative_idx = negative_idx[0]
# # %%
# F_ret = F_ret[:negative_idx]
# time_ret = time_ret[:negative_idx]
#%%
# Determination of Variable
tip = Conical(theta)
E0 = E_0
alpha = 0.42
# %%
Force = F_app_integral(
    t__=time_app,
    E0_=E0,
    alpha_=alpha,
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
#%%
# Curve Fitting(PLR model) Approach Region
F_app_func = partial(
    F_app_integral,
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
popt_app, pcov_app = curve_fit(F_app_func, time_app, F_app, p0=[550, 0.1])
F_app_curvefit = np.array(F_app_func(time_app, *popt_app))
print(popt_app)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time_app, F_app * 1e9, color="red")
ax.plot(time_app, F_app_curvefit * 1e9, color="blue")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
# %%
t1 = Calculation_t1(
    time_ret, t_max, E0, 1e-5, 0.2, velocity_app_func, velocity_ret_func
)
print(t1)
# %%
# Test Retraction Region
tip = Conical(theta)
E0 = popt_app[0]
alpha = popt_app[1]
# time = time_ret

# velocity_ret, indentation_ret은 t1에 해당하는 값을 못먹음(interpolation 범위를 넘어섬)
Force = F_ret_integral(
    t__=t1,
    t___=time_ret,
    E0_=E0,
    alpha_=alpha,
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
Force = np.array(Force)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time_ret, Force * 1e9, color="red")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
# %%
# Curve Fitting(PLR model) Retraction Region
def F_ret_integral_test(
    t__, E0_, alpha_, t_prime_, indentation_, tip_, velocity_app, velocity_ret
):
    t1 = Calculation_t1(t__, t__[0], E0_, t_prime_, alpha_, velocity_app, velocity_ret)
    F = []
    for i, j in zip(t__, t1):
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
        F.append(quad(integrand_, 0, j)[0])
    return F


# %%
F_ret_func = partial(
    F_ret_integral_test,
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
popt_ret, pcov_ret = curve_fit(F_ret_func, time_ret, F_ret, p0=[550, 0.1])
F_ret_curvefit = np.array(F_ret_func(time_ret, *popt_ret))
print(popt_ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time_ret, F_ret * 1e9, color="red")
ax.plot(time_ret, F_ret_curvefit * 1e9, color="blue")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")


# %%
def F_total_integral(
    time, E0_, alpha_, max_ind, t_prime_, indentation_, tip_, velocity_app, velocity_ret
):
    time_app = time[: max_ind + 1]
    time_ret = time[max_ind:]
    t1 = Calculation_t1(
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

    F = F_app[:-1] + F_ret
    return F


# %%
tip = Conical(theta)
length = len(time_app) + len(time_ret) - 1
time = time[:length]
force = force[:length]
force_app_parameter = F_total_integral(
    time=time,
    max_ind=max_ind,
    E0_=popt_app[0],
    alpha_=popt_app[1],
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
force_ret_parameter = F_total_integral(
    time=time,
    max_ind=max_ind,
    E0_=popt_ret[0],
    alpha_=popt_ret[1],
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
force_app_parameter = np.array(force_app_parameter)
force_ret_parameter = np.array(force_ret_parameter)
# %%
F_total_func = partial(
    F_total_integral,
    max_ind=max_ind,
    t_prime_=1e-5,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
# %%
popt_total, pcov_total = curve_fit(F_total_func, time, force, p0=[550, 0.1])
F_total_curvefit = np.array(F_total_func(time, *popt_total))
print(popt_total)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time, F_total_curvefit * 1e9, label="total curvefit")
ax.plot(time, force * 1e9, ".", label="experiment data")
ax.plot(time, force_app_parameter * 1e9, label="approach curvefit")
ax.plot(time, force_ret_parameter * 1e9, label="retraction curvefit")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Force[nN]")
ax.grid(color="lightgrey", linestyle="--")
ax.legend()
# %%
x = ["Approach", "Retraction", "Total"]
modulus = [popt_app[0], popt_ret[0], popt_total[0]]
alpha = [popt_app[1], popt_ret[1], popt_total[1]]
fig, axes = plt.subplots(1, 2, figsize=(10, 7))
axes[0].plot(x, modulus, label="fitting result")
axes[1].plot(x, alpha, label="fitting result")
axes[0].axhline(E_0, color = "r", label = "ground truth")
axes[1].axhline(gamma, color = "r", label = "ground truth")
axes[0].set_title("E0")
axes[1].set_title("α")
axes[0].legend()
axes[1].legend()
axes[0].grid(color="lightgrey", linestyle="--")
axes[1].grid(color="lightgrey", linestyle="--")
# %%
F_at_rp = F_app_integral(t__= time_app, E0_= popt_ret[0], alpha_=popt_ret[1], t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)
F_at_tp = F_app_integral(t__= time_app, E0_= popt_total[0], alpha_=popt_total[1], t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)

t1_rt_ap = Calculation_t1(time_ret, t_max, popt_app[0], 1e-5, popt_app[1], velocity_app_func, velocity_ret_func)
t1_rt_tp = Calculation_t1(time_ret, t_max, popt_total[0], 1e-5,  popt_total[1], velocity_app_func, velocity_ret_func)
F_rt_ap = F_ret_integral(t__ = t1_rt_ap, t___= time_ret, E0_= popt_app[0], alpha_=popt_app[1], t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)
F_rt_tp = F_ret_integral(t__ = t1_rt_tp, t___= time_ret, E0_= popt_total[0], alpha_=popt_total[1], t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)
# %%
MSE_apptime_appparams =np.mean(np.square(F_app_curvefit-F_app))/np.mean(np.square(F_app_curvefit))
MSE_apptime_retparams = np.mean(np.square(F_at_rp-F_app))/np.mean(np.square(F_at_rp))
MSE_apptime_totparams = np.mean(np.square(F_at_tp-F_app))/np.mean(np.square(F_at_tp))

MSE_rettime_appparams = np.mean(np.square(F_rt_ap-F_ret))/np.mean(np.square(F_rt_ap))
MSE_rettime_retparams = np.mean(np.square(F_ret_curvefit-F_ret))/np.mean(np.square(F_ret_curvefit))
MSE_rettime_totparams = np.mean(np.square(F_rt_tp-F_ret))/np.mean(np.square(F_rt_tp))

MSE_tottime_totparams = np.mean(np.square(F_total_curvefit-force))/np.mean(np.square(F_total_curvefit))
MSE_tottime_appparams = np.mean(np.square(force_app_parameter-force))/np.mean(np.square(force_app_parameter))
MSE_tottime_retparams = np.mean(np.square(force_ret_parameter-force))/np.mean(np.square(force_ret_parameter))
# %%
fig,ax = plt.subplots(1, 1, figsize=(7, 5))
tot_time = ["app time - app params", "app time - ret params", "app time - tot params"]
MSE_app = [MSE_apptime_appparams, MSE_apptime_retparams, MSE_apptime_totparams]
ax.bar(tot_time, MSE_app)
ax.set_ylabel("NMSE")
ax.grid(color="lightgray", linestyle='--')
#%%
fig,ax = plt.subplots(1, 1, figsize=(7, 5))
tot_time = ["ret time - app params", "ret time - ret params", "ret time - tot params"]
MSE_ret = [MSE_rettime_appparams, MSE_rettime_retparams, MSE_rettime_totparams]
ax.bar(tot_time, MSE_ret)
ax.set_ylabel("NMSE")
ax.grid(color="lightgray", linestyle='--')
# %%
fig,ax = plt.subplots(1, 1, figsize=(7, 5))
tot_time = ["tot time - app params", "tot time - ret params", "tot time - tot params"]
MSE_tot = [MSE_tottime_appparams, MSE_tottime_retparams, MSE_tottime_totparams]
ax.bar(tot_time, MSE_tot)
ax.set_ylabel("NMSE")
ax.grid(color="lightgray", linestyle='--')
#%%
each_parameter = (
    "approach parameters",
    "retraction parameters",
    "total parameters",
)

each_MSE = {
    "approach time": np.array([MSE_apptime_appparams, MSE_apptime_retparams, MSE_apptime_totparams]),
    "retraction time" : np.array([MSE_rettime_appparams, MSE_rettime_retparams, MSE_rettime_totparams]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(3)

for boolean, weight_count in each_MSE.items():
    p = ax.bar(each_parameter, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

ax.set_title("NMSE of each parameters")
ax.legend(loc="upper right")
ax.set_ylabel("NMSE")
ax.grid(color="lightgray", linestyle='--')
plt.show()
#%%