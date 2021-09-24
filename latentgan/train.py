import os
import torch
import torch.utils.data as data_utils
from latentgan.model import WGAN_GP
from latentgan.dataset import SceneLatentDataset
from pointnet.config import *
from pointnet.model import PointNetVAE

BATCH_SIZE = 64
GENERATOR_ITERS = 100
AE_LOAD_PATH = os.path.join("..", "experiments", model_name, model_params_subdir, epoch_load + ".pth")

model_ae = PointNetVAE()
model_ae.load_state_dict(torch.load(AE_LOAD_PATH))

base_dir = os.path.join("..", data_dir, room_name)
rooms_dir = os.path.join(base_dir, rooms_subdir)

scene_latent_dataset = SceneLatentDataset(rooms_dir, max_num_points, model_ae.encoder)

train_loader = data_utils.DataLoader(
    scene_latent_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_gan = WGAN_GP(batch_size=BATCH_SIZE)
model_gan.train(GENERATOR_ITERS, train_loader)