#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import List
# %%
N_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
N_EPOCHS = 30_000
# %%
key = jax.random.PRNGKey(0)
key, xkey, ynoisekey = jax.random.split(key, 3)
x_samples = jax.random.uniform(xkey, (N_SAMPLES, 1), minval=0.0 , maxval=2*jnp.pi)
y_samples = jnp.sin(x_samples) + jax.random.normal(ynoisekey, (N_SAMPLES, 1)) * 0.3
# %%
plt.scatter(x_samples, y_samples)
# %%
class SimpleMLP(eqx.Module):
    layers: List[eqx.nn.Linear]

    def __init__(self, layer_sizes, key):
        self.layers = []
        for (fan_in, fan_out) in zip(
            layer_sizes[:-1], layer_sizes[1:]
        ):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey)
            )
    def __call__(self, x):
        a = x
        for layer in self.layers[:-1]:
            a = jax.nn.sigmoid(layer(a))
        a = self.layers[-1](a)

        return a
#%%
model = SimpleMLP(LAYERS, key = key)
# %%
# Initial prediction
plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, jax.vmap(model)(x_samples))
# %%
def model_to_loss(m, x, y):
    prediction = jax.vmap(m)(x)
    delta = prediction - y
    return jnp.mean(delta**2)
#%%
model_to_loss(model, x_samples, y_samples)
#%%
opt = optax.sgd(LEARNING_RATE)
opt_state = opt.init(eqx.filter(model, eqx.is_array))
#%%
model_to_loss_and_grad = eqx.filter_value_and_grad(model_to_loss)
#%%
model_to_loss_and_grad(model, x_samples, y_samples)
# %%
@eqx.filter_jit
def make_step(m, opt_s, x, y):
    loss, grad = model_to_loss_and_grad(m, x, y)
    updates, opt_s = opt.update(grad, opt_s, m)
    m = eqx.apply_updates(m, updates)
    return m, opt_s, loss
#%%
loss_history = []
for epoch in range(N_EPOCHS):
    model, opt_state, loss = make_step(model, opt_state, x_samples, y_samples)
    loss_history.append(loss)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch},loss {loss}")
# %%
plt.plot(loss_history)
plt.yscale("log")
# %%
plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, jax.vmap(model)(x_samples))
# %%
