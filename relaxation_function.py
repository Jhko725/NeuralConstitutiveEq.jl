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
#%%
E0 = 573
t_prime = 1e-5
alpha = 0.2
t = np.linspace(0.01, 100, 100)
#%%
plr_t = PLR(t, E0, t_prime, alpha)
fig, ax = plt.subplots(1, 1, figsize =(7,5))
ax.plot(t, plr_t)
# %%
popt, pcov = curve_fit(SLS, t, plr_t)
popt
# %%
sls_t = SLS(t, *popt)
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.plot(t, sls_t)
ax.plot(t, plr_t)
ax.set_xscale("log")
# %%
popt, pcov = curve_fit(KWW, t, plr_t, maxfev=1000000)
popt
# %%
kww_t = KWW(t, *popt)
fig, ax = plt.subplots(1,1, figsize=(7,5))
# ax.plot(t, sls_t)
ax.plot(t, plr_t, '.', label="PLR_consit")
ax.plot(t, kww_t, label='KWW_constit')
ax.plot(t, sls_t, label='SLS_constit')
ax.legend()
# ax.set_xscale("log")
#%%