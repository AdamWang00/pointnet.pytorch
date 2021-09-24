import os
import torch
import torch.utils.data as data_utils
from latentgan.model import WGAN_GP
from latentgan.dataset import SceneLatentDataset
from latentgan.config import *
from pointnetvae.model import PointNetVAE

data_dir = os.path.join("..", data_dir, room_name)
data_rooms_dir = os.path.join(data_dir, rooms_subdir)
ae_load_path = os.path.join("..", ae_model_class, "experiments", ae_model_name, model_params_subdir, ae_epoch_load + ".pth")

model_ae = PointNetVAE()
model_ae.load_state_dict(torch.load(ae_load_path))

scene_latent_dataset = SceneLatentDataset(data_rooms_dir, max_num_points, model_ae.encoder)

BATCH_SIZE = batch_size
NUM_EPOCHS = num_epochs
GENERATOR_ITERS = int(NUM_EPOCHS * scene_latent_dataset.__len__() / BATCH_SIZE / 5)

train_loader = data_utils.DataLoader(
    scene_latent_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_gan = WGAN_GP()
model_gan.train(GENERATOR_ITERS, train_loader)