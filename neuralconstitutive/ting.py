import torch
from torch import Tensor
import pytorch_lightning as pl


class TingApproach(pl.LightningModule):
    def __init__(self, model, tip, lr=0.01):
        super().__init__()
        self.model = model
        self.tip = tip
        self.register_buffer("alpha", torch.tensor(tip.alpha))
        self.register_buffer("beta", torch.tensor(tip.beta))
        self.loss_func = torch.nn.MSELoss()
        self.lr = lr

    def stress_relaxation(self, t: Tensor):
        return self.model(t.view(-1, 1)).view(-1)

    def forward(self, t: Tensor, v: Tensor, I: Tensor):
        phi = self.stress_relaxation(t)
        dI_beta = v * I ** (self.beta - 1)

        def _inner(ind: int):
            phi_ = torch.flip(phi[0 : ind + 1], dims=(0,))
            t_ = t[0 : ind + 1]
            dI_beta_ = dI_beta[0 : ind + 1]
            return torch.trapz(phi_ * dI_beta_, x=t_).view(-1)

        f = torch.cat([_inner(i) for i in range(len(t))], dim=0)
        return self.alpha * f

    def training_step(self, batch, batch_idx):
        t, v, I, f_true = (
            batch["time"].view(-1),
            batch["velocity"].view(-1),
            batch["indent"].view(-1),
            batch["force"].view(-1),
        )
        f_pred = self(t, v, I)
        train_loss = self.loss_func(f_true, f_pred)
        self.log("train_loss", train_loss)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
