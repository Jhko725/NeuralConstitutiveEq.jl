#%%
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar

#%%
t = np.linspace(-10, 10, 1000)
a = gammainc(5, t)
plt.plot(t, a)
#%%
def F_sls_app(t, E0, E_inf, tau, tip):
    a = tip.alpha 
    b = tip.beta
    gamma(b)*gamma(b, )
# %%
