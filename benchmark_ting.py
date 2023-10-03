# %%
from typing import Callable
import functools

import numpy as np
from numpy import ndarray
from scipy.integrate import quad
from scipy.optimize import root_scalar

from neuralconstitutive.tipgeometry import TipGeometry
from neuralconstitutive.constitutive import PowerLawRheology
from neuralconstitutive.ting import force_approach, force_retract

plr = PowerLawRheology(1.0, 2.0, 3.0)
plr2 = type(plr)(1.0, 3.0, 4.0)
plr2
#%%

#%%
def force_approach2(
    t: ndarray,
    constit: Callable[[ndarray], ndarray],
    indent_app: ndarray,
    velocity_app: ndarray,
    tip: TipGeometry,
    **quad_kwargs,
) -> ndarray:
    dF = make_force_integand(constit, velocity_app, indent_app, tip)
    F = np.stack([quad(dF, 0, t_i, args=(t_i,), **quad_kwargs)[0] for t_i in t], axis=0)
    return F

def integrate(y, x, x_lower, x_upper):
    
#%%
plr = PowerLawRheology(0.572, 0.42, 1e-5)
t = np.linspace(0, 0.4, 200)
#%%