import torch
from pointnet.model import PointNetVAE
from pointnet.config import *

LOAD_PATH = "model1/9.pth"
model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

print(model.generate())