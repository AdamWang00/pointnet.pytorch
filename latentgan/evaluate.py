import os
import torch
import torch.utils.data as data_utils
import numpy as np
from scipy.linalg import sqrtm
from latentgan.config import *
from latentgan.dataset import SceneLatentDataset
from latentgan.model import WGAN_GP
from pointnetvae.model import PointNetVAE
from pointnetae.model import PointNetAE

NUM_FAKE_CODES = 10000 # 1000000
EVALUATE_VAE = False

ae_reference_model_class = "pointnetvae"
ae_reference_model_name = "bedroom_full1"
ae_reference_epoch_load = "latest"

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

ae_load_path = os.path.join("..", ae_model_class, "experiments", ae_model_name, model_params_subdir, ae_epoch_load + ".pth")
if ae_model_class == "pointnetvae":
    model_ae = PointNetVAE()
elif ae_model_class == "pointnetae":
    model_ae = PointNetAE()
model_ae.load_state_dict(torch.load(ae_load_path))
model_ae.eval()

ae_reference_load_path = os.path.join("..", ae_reference_model_class, "experiments", ae_reference_model_name, model_params_subdir, ae_reference_epoch_load + ".pth")
if ae_reference_model_class == "pointnetvae":
    model_ae_reference = PointNetVAE()
elif ae_reference_model_class == "pointnetae":
    model_ae_reference = PointNetAE()
model_ae_reference.load_state_dict(torch.load(ae_reference_load_path))
model_ae_reference.eval()

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

real_latent_codes = train_loader._get_iterator().next().cuda().detach()
model_ae.cuda()
model_ae_reference.cuda()
real_generated_scenes = model_ae.generate(latent_code=real_latent_codes).detach()
real_latent_codes_reference = model_ae_reference.encoder(real_generated_scenes.transpose(2, 1))[0].detach()
real_latent_codes = real_latent_codes.cpu().numpy()
real_latent_codes_reference = real_latent_codes_reference.cpu().numpy()

if EVALUATE_VAE:
    # OUTDATED
    fake_latent_codes = torch.randn(NUM_FAKE_CODES, latent_size)
    print(calculate_frechet(real_latent_codes.numpy(), fake_latent_codes.numpy()))
else:

    # TODO: normalize fake_generated_scenes such that it resembles the input to model_ae_reference.encoder (e.g. replace boxes with existence < 0 with zeros, make category one-hot, normalize and clip orientation; see generate.py)

    gan_load_path = os.path.join("experiments", model_name, model_params_subdir)
    model_gan = WGAN_GP()
    if not iter_load.isdigit():
        model_gan.load_model(gan_load_path, iter)
        model_gan.eval()

        fake_latent_codes = model_gan.generate(num_codes=NUM_FAKE_CODES).detach()
        fake_generated_scenes = model_ae.generate(latent_code=fake_latent_codes).detach()
        fake_latent_codes_reference = model_ae_reference.encoder(fake_generated_scenes.transpose(2, 1))[0].detach()

        fake_latent_codes = fake_latent_codes.cpu().numpy()
        fake_latent_codes_reference = fake_latent_codes_reference.cpu().numpy()
        frechet = calculate_frechet(real_latent_codes, fake_latent_codes)
        frechet_reference = calculate_frechet(real_latent_codes_reference, fake_latent_codes_reference)
        print("ITER", iter, frechet, frechet_reference)
    else:
        for iter in range(save_per_iters, int(iter_load) + save_per_iters, save_per_iters):
            model_gan.load_model(gan_load_path, iter)
            model_gan.eval()

            fake_latent_codes = model_gan.generate(num_codes=NUM_FAKE_CODES).detach()
            fake_generated_scenes = model_ae.generate(latent_code=fake_latent_codes).detach()
            fake_latent_codes_reference = model_ae_reference.encoder(fake_generated_scenes.transpose(2, 1))[0].detach()

            fake_latent_codes = fake_latent_codes.cpu().numpy()
            fake_latent_codes_reference = fake_latent_codes_reference.cpu().numpy()
            frechet = calculate_frechet(real_latent_codes, fake_latent_codes)
            frechet_reference = calculate_frechet(real_latent_codes_reference, fake_latent_codes_reference)
            print("ITER", iter, frechet, frechet_reference)