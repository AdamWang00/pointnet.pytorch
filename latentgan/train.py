import os
import torch
import torch.utils.data as data_utils
from latentgan.model import WGAN_GP
from latentgan.dataset import SceneLatentDataset
from latentgan.config import *
if ae_model_class == "pointnetvae":
    from pointnetvae.model import PointNetVAE
elif ae_model_class == "pointnetae":
    from pointnetae.model import PointNetAE

data_dir = os.path.join(data_dir, room_name)
data_rooms_dir = os.path.join(data_dir, rooms_subdir)
ae_load_path = os.path.join("..", ae_model_class, "experiments", ae_model_name, model_params_subdir, ae_epoch_load + ".pth")

if ae_model_class == "pointnetvae":
    model_ae = PointNetVAE()
elif ae_model_class == "pointnetae":
    model_ae = PointNetAE()
model_ae.load_state_dict(torch.load(ae_load_path))

scene_latent_dataset = SceneLatentDataset(data_rooms_dir, max_num_points, model_ae.encoder, load_ram=True)

BATCH_SIZE = batch_size
GENERATOR_ITERS = generator_iters # int(NUM_EPOCHS * scene_latent_dataset.__len__() / BATCH_SIZE / 5)

train_loader = data_utils.DataLoader(
    scene_latent_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_gan = WGAN_GP()
model_gan.train(GENERATOR_ITERS, train_loader)