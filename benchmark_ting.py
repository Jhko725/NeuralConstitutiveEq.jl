# %%
from typing import Callable
import functools

import numpy as np
from numpy import ndarray
from scipy.integrate import quad
from scipy.optimize import root_scalar

from neuralconstitutive.tipgeometry import TipGeometry
from neuralconstitutive.ting import force_approach, force_retract
