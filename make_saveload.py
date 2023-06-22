# %%
import torch
from neuralconstitutive.ting import TingApproach
from neuralconstitutive.tipgeometry import Conical
from neuralconstitutive.models import FullyConnectedNetwork, BernsteinNN

ckpt_path = "hydrogel/w1cpp7ha/checkpoints/epoch=9999-step=10000.ckpt"
tip = Conical(torch.pi / 36)
model = FullyConnectedNetwork([1, 20, 20, 20, 1], torch.nn.functional.elu)
ting = TingApproach.load_from_checkpoint(
    checkpoint_path=ckpt_path, model=BernsteinNN(model, 100), tip=tip
)
# %%
