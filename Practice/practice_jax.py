#%%
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
from more_itertools import pairwise
import scipy
import matplotlib.pyplot as plt
# %%
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
    
    def __call__(self, x):
        return self.weight @ x + self.bias
#%%
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y-pred_y)**2)

batch_size, in_size, out_size = 32, 2, 3
model = Linear(in_size, out_size, key = jax.random.PRNGKey(0))
x = jax.numpy.zeros((batch_size, in_size))
y = jax.numpy.zeros((batch_size, out_size))
grads = loss_fn(model, x, y)
# %%
model(jnp.array([2,1]))
# %%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import List

IN_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
IN_EPOCHS = 30_000

key = jax.random.PRNGKey(0)
key, xkey, ynoisekey = jax.random.split(key, 3)
print(key, xkey, ynoisekey)
# %%
