import numpy as np
from numpy import ndarray


def squared_error(data: ndarray, target: ndarray) -> float:
    return np.sum((data - target) ** 2)
