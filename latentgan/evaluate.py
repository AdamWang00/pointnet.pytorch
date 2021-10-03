import os
import torch
import torch.utils.data as data_utils
import numpy as np
from scipy.linalg import sqrtm
from latentgan.config import *
from latentgan.dataset import SceneLatentDataset
from latentgan.model import WGAN_GP
from pointnetvae.model import PointNetVAE

NUM_FAKE_CODES = 1000000
EVALUATE_VAE = False

# implementation from https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
def calculate_frechet(samp1, samp2):
	# calculate mean and covariance statistics
	mu1, sigma1 = np.mean(samp1, axis=0), np.cov(samp1, rowvar=False)
	mu2, sigma2 = np.mean(samp2, axis=0), np.cov(samp2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

if not EVALUATE_VAE:
    gan_load_path = os.path.join("experiments", model_name, model_params_subdir)
    model_gan = WGAN_GP()
    model_gan.load_model(gan_load_path, epoch_load)
    model_gan.eval()

ae_load_path = os.path.join("..", ae_model_class, "experiments", ae_model_name, model_params_subdir, ae_epoch_load + ".pth")
model_ae = PointNetVAE()
model_ae.load_state_dict(torch.load(ae_load_path))

data_dir = os.path.join(data_dir, room_name)
data_rooms_dir = os.path.join(data_dir, rooms_subdir)
scene_latent_dataset = SceneLatentDataset(data_rooms_dir, max_num_points, model_ae.encoder)

train_loader = data_utils.DataLoader(
    scene_latent_dataset,
    batch_size=scene_latent_dataset.__len__(),
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

real_latent_codes = train_loader._get_iterator().next()
if EVALUATE_VAE:
    fake_latent_codes = torch.randn(NUM_FAKE_CODES, latent_size)
else:
    fake_latent_codes = model_gan.generate(num_codes=NUM_FAKE_CODES).cpu()

print(calculate_frechet(real_latent_codes.numpy(), fake_latent_codes.numpy()))