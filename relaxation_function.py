#%%
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
def PLR(t, E0, t_prime, alpha):
    return E0*(1+t/t_prime)**(-alpha)

def SLS(t, E_inf, E0, tau):
    return E_inf+(E0-E_inf)*np.exp(-(t/tau))

def KWW(t, E_inf, E0, tau, beta):
    return E_inf + (E0-E_inf)*np.exp(-(t/tau)**beta)

def Fung(t, C, E0, Ei, tau1, tau2):
    return E0*((1+C*(Ei*(t/tau2)-Ei*(t/tau1)))/(1+C*np.log(tau2/tau1)))
#%%
E0 = 40
t_prime = 1e-5
alpha = 0.4
t = np.linspace(0.01, 100, 100)
#%%
plr_t = PLR(t, E0, t_prime, alpha)
fig, ax = plt.subplots(1, 1, figsize =(7,5))
ax.plot(t, plr_t)
# %%
popt_sls, pcov_sls = curve_fit(SLS, t, plr_t)
popt_sls
# %%
sls_t = SLS(t, *popt_sls)
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.plot(t, sls_t)
ax.plot(t, plr_t)
ax.set_xscale("log")
# %%
popt_kww, pcov_kww = curve_fit(KWW, t, plr_t, maxfev=1000000)
popt_kww
# %%
kww_t = KWW(t, *popt_kww)
#%%
popt_fung, pcov_fung = curve_fit(Fung, t, plr_t)
popt_fung
# %%
fung_t = Fung(t, *popt_fung)

fig, ax = plt.subplots(1,1, figsize=(7,5))
# ax.plot(t, sls_t)
ax.plot(t, plr_t, '.', label="PLR_consit")
ax.plot(t, kww_t, label='KWW_constit')
ax.plot(t, sls_t, label='SLS_constit')
ax.plot(t, fung_t, label="Fung_constit")
ax.legend()
# ax.set_xscale("log")
# ax.set_yscale("log")
# %%
