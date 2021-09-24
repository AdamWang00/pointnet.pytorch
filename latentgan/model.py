import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import os
from pointnet.config import *

SAVE_PER_TIMES = 100


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        return self.main_module(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main_module(x)


class WGAN_GP(object):
    def __init__(self, batch_size=None):
        self.G = Generator().cuda()
        self.D = Discriminator().cuda()

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        if batch_size == None:
            self.batch_size = 64
        else:
            self.batch_size = batch_size

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.critic_iter = 5
        self.lambda_term = 10


    def train(self, generator_iters, train_loader):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        for g_iter in range(generator_iters):
            # Train Discriminator
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                real_latent_codes = self.data.__next__().cuda()

                d_loss_real = self.D(real_latent_codes)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                z = torch.randn(self.batch_size, 64).cuda()

                fake_latent_codes = self.G(z)
                d_loss_fake = self.D(fake_latent_codes)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                gradient_penalty = self.calculate_gradient_penalty(real_latent_codes, fake_latent_codes)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                # Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter + 1}/{self.critic_iter}, loss: {d_loss}')

            # Train Generator
            for p in self.D.parameters():
                p.requires_grad = False # to avoid update

            self.G.zero_grad()

            z = torch.randn(self.batch_size, 64).cuda()

            fake_latent_codes = self.G(z)
            g_loss = self.D(fake_latent_codes)
            g_loss = g_loss.mean()
            g_loss.backward(mone)

            self.g_optimizer.step()

            print(f'Generator iteration: {g_iter + 1}/{generator_iters}, g_loss: {g_loss}')

            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()

        self.t_end = t.time()
        print('Time of training: {}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model()


    def generate(self, z=None):
        if z == None:
            z = torch.randn(1, 64)
        samples = self.G(z)
        return samples


    def calculate_gradient_penalty(self, real_latent_codes, fake_latent_codes):
        eta = torch.FloatTensor(self.batch_size,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_latent_codes.size(1))
        eta = eta.cuda()

        interpolated = eta * real_latent_codes + ((1 - eta) * fake_latent_codes)

        interpolated = interpolated.cuda()

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty


    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')


    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))


    def get_infinite_batches(self, data_loader):
        while True:
            for real_latent_codes in data_loader:
                yield real_latent_codes


    def eval(self):
        self.D.eval()
        self.G.eval()

    # def generate_latent_walk(self, number):
    #     if not os.path.exists('interpolated_images/'):
    #         os.makedirs('interpolated_images/')

    #     number_int = 10
    #     # interpolate between twe noise(z1, z2).
    #     z_intp = torch.FloatTensor(1, 100, 1, 1)
    #     z1 = torch.randn(1, 100, 1, 1)
    #     z2 = torch.randn(1, 100, 1, 1)
    #     if self.cuda:
    #         z_intp = z_intp.cuda()
    #         z1 = z1.cuda()
    #         z2 = z2.cuda()

    #     z_intp = Variable(z_intp)
    #     real_latent_codes = []
    #     alpha = 1.0 / float(number_int + 1)
    #     print(alpha)
    #     for i in range(1, number_int + 1):
    #         z_intp.data = z1*alpha + z2*(1.0 - alpha)
    #         alpha += alpha
    #         fake_im = self.G(z_intp)
    #         fake_im = fake_im.mul(0.5).add(0.5) #denormalize
    #         real_latent_codes.append(fake_im.view(self.C,32,32).data.cpu())

    #     grid = utils.make_grid(real_latent_codes, nrow=number_int )
    #     utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
    #     print("Saved interpolated real_latent_codes.")