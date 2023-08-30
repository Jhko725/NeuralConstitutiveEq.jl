#%%
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from random import *
from neuralconstitutive import utils
import math
#%%
E = 0.572 # 572Pa
t_0 = 1 # 1s
gamma = torch.Tensor([0.2])
theta = torch.pi/10.0 # Conical
a = 8.0/(3.0*torch.pi)*np.tan(theta)
b = torch.Tensor([2.0])
v = 10.0 # 10um/s
#%%
coeff = E*t_0**gamma*a*b*v**b
coeff = coeff*utils.beta(b, 1.0-gamma)
print(coeff)
#%%
def PLR_Force(time, coeff, b, gamma):
    return coeff*time**(b-gamma)

def t1(time, t_max):
    coeff = 2.0**(1.0/(1.0-gamma))
    return torch.clamp(time-coeff*(time-t_max), 0.0, None)

#%%
t_array = torch.arange(0.0, 0.401, 0.001)
t_array_app = torch.arange(0.0, 0.201, 0.001)
t_max = (t_array[0]+t_array[len(t_array)-1])/2
print(t_array)
print(len(t_array_app))
plr_force = PLR_Force(t_array, coeff, b, gamma)
#%%
total = []
ret = []
for i in t_array:
    if i <= t_max:
        total.append(float(PLR_Force(i, coeff, b, gamma)))
    else :
        ret.append(float(PLR_Force(t1(i, t_max), coeff, b, gamma)))
# for i in ret:
#     total.append(i)
total = torch.Tensor(total)
print(len(total))
print(len(t_array))
#%%
noise_amp = 0.03
noise = torch.randn(len(total))*noise_amp
# total_noise = total + noise
#%%
fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
ax.set_title("PLR model simulation")
ax.plot(t_array_app, total)
ax.scatter(t_array_app, total+noise, c="orange", s=20, alpha=1)
#%%
F_app = total
I = t_array_app * v

F_app = F_app.unsqueeze(1)
I = I.unsqueeze(1)

print(F_app)
F_app.size()

# %%
linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(
    linear_model.parameters(),
    lr=1e-2)
# %%
linear_model.parameters()
# %%
list(linear_model.parameters())
# %%
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):

        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10000 == 0 :
            print(f"Epoch {epoch}, Training loss {loss_train.item() : .4f}," f"Validation loss {loss_val.item() : .4f}")
#%%
linear_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)

optimizer = optim.SGD(linear_model.parameters(), lr=1e-4)
t_u_train, t_u_val, t_c_train, t_c_val = train_test_split(F_app, I, test_size=0.1, random_state=1)
#%%
training_loop(
    n_epochs = 2200000,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_u_train,
    t_u_val = t_u_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)

# %%
list(linear_model.parameters())
# %%
