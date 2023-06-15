# %%
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from NeuralNetwork.utils import beta
from NeuralNetwork.dataset import IndentationDataset, split_app_ret
from NeuralNetwork.models import FullyConnectedNetwork


def simulate_PLR(E0, gamma, t0, t_max, dt):
    def t1(time):
        coeff = 2.0 ** (1.0 / (1.0 - gamma))
        return torch.clamp(time - coeff * (time - t_max), 0.0, None)

    theta = torch.tensor(torch.pi / 10.0)  # Conical
    a = 8.0 / (3.0 * torch.pi) * torch.tan(theta)
    b = torch.tensor(2.0)
    v = 10.0  # 10um/s
    gamma = torch.tensor(gamma)
    coeff = E0 * t0**gamma * a * b * v**b * beta(b, 1.0 - gamma)
    time = torch.arange(0.0, 2 * t_max, dt)
    is_app = time <= t_max
    time_1 = torch.cat([time[is_app], t1(time[~is_app])], axis=-1)
    force = coeff * time_1 ** (b - gamma)
    indent = torch.cat([v * time[is_app], 2 * v * t_max - v * time[~is_app]], axis=-1)
    return IndentationDataset(time.view(1, -1), indent.view(1, -1), force.view(1, -1))


dataset = simulate_PLR(0.572, 0.42, 1.0, 0.2, 0.001)

# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 4), constrained_layout=True)
axes[0].plot(dataset.indent[0], dataset.force[0])
axes[1].plot(dataset.time[0], dataset.indent[0])
axes[2].plot(dataset.time[0], dataset.force[0])


# %%
dataset_app, dataset_ret = split_app_ret(dataset)
# %%
print(dataset_app.time.shape)
# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 4), constrained_layout=True)
axes[0].plot(dataset_app.indent, dataset_app.force)
axes[1].plot(dataset_app.time, dataset_app.indent)
axes[2].plot(dataset_app.time, dataset_app.force)
axes[0].plot(dataset_ret.indent, dataset_ret.force)
axes[1].plot(dataset_ret.time, dataset_ret.indent)
axes[2].plot(dataset_ret.time, dataset_ret.force)


# %%
class TingApproach(pl.LightningModule):
    def __init__(self, model, time, indent, lr=0.01):
        super().__init__()
        self.model = model
        self.register_buffer("time", time)
        self.register_buffer("dI_beta", 10.0 * indent)
        self.loss_func = torch.nn.MSELoss()
        self.lr = lr

    def integrand(self, t):
        inds = self.time <= t
        time_ = self.time[inds]
        y = self.model(t - time_.view(-1, 1)) * self.dI_beta[inds].view(-1, 1)
        return torch.trapezoid(y.view(-1), x=time_)

    def forward(self, t):
        return torch.cat([self.integrand(t_).view(-1) for t_ in t])

    def training_step(self, batch, batch_idx):
        t, f_true = batch["time"].view(-1), batch["force"].view(-1)
        f_pred = self(t)
        train_loss = self.loss_func(f_true, f_pred)
        self.log("train_loss", train_loss)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# %%
def PLR(t):
    theta = torch.tensor(torch.pi / 10.0)
    a = 8.0 / (3.0 * torch.pi) * torch.tan(theta)
    return a * 0.572 * t ** (-0.42)


model_true = TingApproach(PLR, dataset_app.time, dataset_app.indent, lr=1e-3)
with torch.no_grad():
    f_pred = model_true(dataset_app.time)
    print(f_pred.shape)
# %%
# logger = WandbLogger(project="hydrogel", entity="jhelab")

model = FullyConnectedNetwork([1, 10, 10, 1], torch.nn.functional.relu)
ting = TingApproach(model, dataset_app.time, dataset_app.indent, lr=1e-3)
# %%
with torch.no_grad():
    f_pred = ting(dataset_app.time)
    print(f_pred.shape)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(dataset_app.time, dataset_app.force, label="Data")
axes[0].plot(dataset_app.time, f_pred, label="NN_untrained")
axes[0].legend()
with torch.no_grad():
    axes[1].plot(dataset_app.time, model(dataset_app.time.view(-1, 1)), ".")
# %%
dataloader = torch.utils.data.DataLoader(
    dataset_app,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
)
trainer = pl.Trainer(
    max_epochs=1000,
    log_every_n_steps=1,
    deterministic="warn",
    accelerator="gpu",
    devices=1,
    logger=logger,
)
trainer.fit(ting, dataloader)
# %%
with torch.no_grad():
    f_pred = ting(dataset_app.time.view(-1))
    print(f_pred.shape)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(dataset_app.time.view(-1), dataset_app.force.view(-1), label="Data")
axes[0].plot(dataset_app.time.view(-1), f_pred.view(-1), label="NN_untrained")
axes[0].legend()
with torch.no_grad():
    axes[1].plot(dataset_app.time.view(-1), model(dataset_app.time.view(-1, 1)), ".")
# %%
