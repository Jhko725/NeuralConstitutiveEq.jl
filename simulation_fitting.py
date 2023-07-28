#%%
import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from neuralconstitutive import utils
# %%
E_0 = 572 # Pa
gamma = 0.42
t_0 = 1.0 # s
v = 10*1e-6 # m/s
theta = (18.0/180.0)*np.pi
alpha = 8.0/(3.0*np.pi) *np.tan(theta)
beta = 2.0
#%%
def PLR_Force(E_0, gamma, t, t_0, v, alpha, beta):
    force = E_0*alpha*beta*(v**beta)*(t**beta)*(((t/t_0))**(-gamma))*sc.beta(beta, 1-gamma)
    return np.array(force)
#%%
def t1(t, t_max, gamma):
    t1 = t - 2**(1.0/(1.0-gamma))*(t-t_max)
    t1 = np.clip(t1, 0, np.inf)
    return t1
#%%
space = 200
idx = int(space/2)
t_array = np.linspace(0, 4, space)
t_app = t_array[:idx]
t_ret = t_array[idx:]
t_max = (t_array[idx]+t_array[idx+1])/2
t_1 = t1(t_ret, t_max, gamma)
# %%
F_app = PLR_Force(E_0, gamma, t_app, t_0, v, alpha, beta)
F_ret = PLR_Force(E_0, gamma, t_1, t_0, v, alpha, beta)
F_ret[np.isnan(F_ret)] = 0
# %%
F_total = np.append(F_app, F_ret)
#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(t_array, F_total * 1e9, '.')
# %%
