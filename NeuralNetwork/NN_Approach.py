#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from random import *
import utils
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
# %%
# Generate PLR model Simulation data
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
t_max = (t_array[0]+t_array[len(t_array)-1])/2
print(t_array)

plr_force = PLR_Force(t_array, coeff, b, gamma)
#%%
total = []
ret = []
for i in t_array:
    if i < t_max:
        total.append(float(PLR_Force(i, coeff, b, gamma)))
    else :
        ret.append(float(PLR_Force(t1(i, t_max), coeff, b, gamma)))
for i in ret:
    total.append(i)
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
ax.plot(t_array, total)
ax.scatter(t_array, total+noise, c="orange", s=20, alpha=1)

#%%
t_array = torch.arange(0.0, 0.201, 0.001)
# t_max = (t_array[0]+t_array[len(t_array)-1])/2
t_max = 0.2
print(t_array)

plr_force = PLR_Force(t_array, coeff, b, gamma)
#%%
total = []
ret = []
for i in t_array:
    if i < t_max:
        total.append(float(PLR_Force(i, coeff, b, gamma)))
    else :
        ret.append(float(PLR_Force(t1(i, t_max), coeff, b, gamma)))
for i in ret:
    total.append(i)
total = torch.Tensor(total)
print(len(total))
print(len(t_array))
#%%
noise_amp = 0.04
noise = torch.randn(len(total))*noise_amp
# total_noise = total + noise
#%%
train_data = total+noise
test_data = total
train_data.unsqueeze(1)
test_data.unsqueeze(1)
#%%
# NeuralNetwork
model = torch.nn.Sequential(
    torch.nn.Linear(201, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6

for t in range(10000):
    y_pred = model(train_data)
    loss = loss_fn(y_pred, test_data)
    if t % 1000 == 1000:
        print(f"{t}th, loss = {loss.item()}")

model.zero_grad()
loss.backward()

with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate*param.grad

linear_layer = model[0]
print(list(linear_layer.prameters()))
print(linear_layer.weight[:,:].size())

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2]} x^3')
# %%
input = torch.randn(32, 1, 5, 5)
# With default parameters
m = nn.Flatten()
output = m(input)
print(output.size())
# With non-default parameters
m = nn.Flatten(0, 2)
output = m(input)
print(output.size())
# %%
x = torch.tensor([[1,2],[3,4],[5,6]])
y = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(y)
x.size()
y.size()
xx.size()

xx
# %%
