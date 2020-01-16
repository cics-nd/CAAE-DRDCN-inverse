import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR

import torch.nn as nn
import torch.nn.functional as F
import torch
from data_plot3D import plot_pred, load_data
from CAAE3D_models import Encoder, Decoder, Discriminator
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')

os.makedirs("images_CAAE", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--current-dir", type=str, default="/afs/crc.nd.edu/user/s/smo/CAAE3D/", help="data directory")
parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument('--n-train', type=int, default=37600, help='number of training data')
parser.add_argument('--n-test', type=int, default=3000, help='number of training data')
parser.add_argument("--batch-size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lw", type=float, default=0.01, help="adversarial loss weight")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample-interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

date = 'experiments/Dec_26_CAAE3D'
exp_dir = opt.current_dir + date + "/N{}_Bts{}_Eps{}_lr{}_lw{}".\
    format(opt.n_train, opt.batch_size, opt.n_epochs, opt.lr, opt.lw)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# loss functions
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

nf, d, h, w = 2, 2, 8, 16

# Initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)
discriminator = Discriminator(inchannels=nf)

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()+discriminator._n_parameters()))

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

hdf5_dir = "/afs/crc.nd.edu/user/s/smo/multi-fidelity-CNN/raw2/3D/"
dataloader, x_test = load_data(hdf5_dir, opt)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def test(epoch):
    encoder.eval()
    decoder.eval()

    n_samples = 1
    z = Variable(Tensor(np.random.normal(0, 1, (n_samples, nf, d, h, w))))
    gen_imgs = decoder(z)
    samples = np.squeeze(gen_imgs.data.cpu().numpy())
    plot_pred(samples,epoch,0,output_dir)

    idx = np.random.choice(opt.n_test, 1, replace=False)
    real_imgs = x_test[[idx]]
    real_imgs = (torch.FloatTensor(real_imgs)).cuda()
    encoded_imgs = encoder(real_imgs)
    decoded_imgs = decoder(encoded_imgs)
    samples_gen  = np.squeeze(decoded_imgs.data.cpu().numpy())
    samples_real = np.squeeze(real_imgs.data.cpu().numpy())

    samples = np.vstack((samples_real[:3],samples_gen[:3],samples_real[3:],samples_gen[3:]))
    plot_pred(samples,epoch+1,idx,output_dir)


# ----------
#  Training
# ----------
for epoch in range(1,opt.n_epochs+1):
    encoder.train()
    decoder.train()
    discriminator.train()

    for i, (imgs,) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0],1).fill_(1.0), requires_grad=False)
        fake  = Variable(Tensor(imgs.shape[0],1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss_a = adversarial_loss(discriminator(encoded_imgs), valid)
        g_loss_c = pixelwise_loss(decoded_imgs, real_imgs)

        g_loss = opt.lw * g_loss_a + g_loss_c

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], nf, d, h, w))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    batches_done = epoch * len(dataloader) + i
    if (epoch) % opt.sample_interval == 0:
        test(epoch)

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f /G_A loss: %f/ G_C loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), g_loss_a.item(), g_loss_c.item())
    )

torch.save(decoder.state_dict(), model_dir + '/AAE_decoder_epoch{}.pth'.format(opt.n_epochs))
torch.save(encoder.state_dict(), model_dir + '/AAE_encoder_epoch{}.pth'.format(opt.n_epochs))
torch.save(discriminator.state_dict(), model_dir + '/AAE_discriminator_epoch{}.pth'.format(opt.n_epochs))
