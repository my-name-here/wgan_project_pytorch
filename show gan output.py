#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:52:22 2021

@author: christopherbrower
"""

#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

"""
An implementation of wgan, hacked together from various tutorials.

using
https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
 for the gan models, and main training script, and a modified version of
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
to load images
uses https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
for wgan loss and when to train

https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
for changes to wgan

I also used
https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_clipping.py
for the weight clippign
for loading images with transparency, use https://github.com/pytorch/vision/issues/2276


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.utils as vutils

"""


# In[37]:



import errno
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython import display
from PIL import Image

model_name = "wgan"
data_name = "images"
def transparency_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA")


# do images hve transparency
transparency = False
# should the image generation be grayscale?
grayscale = True
# what epoch to load from
load_from = 63
# is cuda availabe
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
# Root directory for dataset, using faces set, as that is the folder
dataroot = "./"
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Number of channels in the training images. For color images this is 3
if grayscale:
    nc = 1
else:
    nc = 3

# Batch size during training
batch_size = 64
# Number of workers for dataloader
workers = 0
# number of epochs to train for
num_epochs = 200
# should it use weight clipping
weight_clipping = True
clipping_value = 0.01
# We can use an image folder dataset the way we have it setup.
# Create the dataset usinmge custom loader if transsparent images, and default otherwise
if transparency:
    # add a transparency channel, by increasing number of channels by one
    nc += 1
    # this doesn't actually work,
    if grayscale:
        dataset = datasets.ImageFolder(
            root=dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        tuple(0.5 for i in range(nc)), tuple(0.5 for i in range(nc))
                    ),
                    transforms.Grayscale(num_output_channels=nc),
                ]
            ),
            loader=transparency_loader,
        )
    else:
        dataset = datasets.ImageFolder(
            root=dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        tuple(0.5 for i in range(nc)), tuple(0.5 for i in range(nc))
                    ),
                ]
            ),
            loader=transparency_loader,
        )
else:
    if grayscale:
        dataset = datasets.ImageFolder(
            root=dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Grayscale(num_output_channels=1),
                ]
            ),
        )
    else:
        dataset = datasets.ImageFolder(
            root=dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    # uncomment the next line for grayscale when channels is 1
                    # ,transforms.Grayscale(num_output_channels=1)
                ]
            ),
        )

# Create the dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64], padding=2, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)

img_shape = (nc, image_size, image_size)
# the size of the noise
z_dim = 100
# Learning rate for optimizers
lr = 0.0002

# sample_interval is how often we check images
sample_interval = 1
num_batches = len(dataloader)
# the number of discriminator training steps before a generator training step, for wgan
n_critic = 5


# In[38]:

# gives us information about the training progress,
# by telling us losses, and showing images

comment = "{}_{}".format(model_name, data_name)
data_subdir = "{}/{}".format(model_name, data_name)
def _step(epoch, n_batch, num_batches):
    return epoch * num_batches + n_batch

def _make_dir(directory):
    """
    Make a directory at the given path

    Parameters
    ----------
    directory : string
        The path to make the directory at.

    Returns
    -------
    None.

    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def log(d_error, g_error, epoch, n_batch, num_batches):

   # var_class = torch.autograd.variable.Variable
   if isinstance(d_error, torch.autograd.Variable):
       d_error = d_error.data.cpu().numpy()
   if isinstance(g_error, torch.autograd.Variable):
       g_error = g_error.data.cpu().numpy()

   step =_step(epoch, n_batch, num_batches)
  



def display_status(epoch, num_epochs,n_batch,num_batches,d_error,g_error,d_pred_real,d_pred_fake,):
    """
    Display some information about training.

    Parameters
    ----------
    epoch : int
        the epoch you are on.
    num_epochs : int
        the number of epochs ytou train for.
    n_batch : int
        the batch you are on.
    num_batches : int
        the number of batches per epoch, this is dataset_size/batch_size, or len(dataloader.
    d_error : float
        the error of the discriminator.
    g_error : float
        the error of the generator.
    d_pred_real : float
        what the average prediction is for a real image.
    d_pred_fake : float
        what the avverage prediction is for a fake image.

    Returns
    -------
    None.

    """

    # var_class = torch.autograd.variable.Variable
    if isinstance(d_error, torch.autograd.Variable):
        d_error = d_error.data.cpu().numpy()
    if isinstance(g_error, torch.autograd.Variable):
        g_error = g_error.data.cpu().numpy()
    if isinstance(d_pred_real, torch.autograd.Variable):
        d_pred_real = d_pred_real.data
    if isinstance(d_pred_fake, torch.autograd.Variable):
        d_pred_fake = d_pred_fake.data

    print(
        "Epoch: [{}/{}], Batch Num: [{}/{}]".format(
            epoch, num_epochs, n_batch, num_batches
        )
    )
    print(
        "Discriminator Loss: {:.4f}, Generator Loss: {:.4f}".format(
            d_error, g_error
        )
    )
    print(
        "D(x): {:.4f}, D(G(z)): {:.4f}".format(
            d_pred_real.mean(), d_pred_fake.mean()
        )
    )
   
def save_models(generator, discriminator, epoch):
    """
    Save your models at a given epoch.

    Parameters
    ----------
    generator : GeneratorNet
        The Generator model to save.
    discriminator : DiscriminatorNet
        The Discriminator model to save.
    epoch : int
        The epoch you are on.

    Returns
    -------
    None.

    """
    out_dir = "./data/models/{}".format(data_subdir)
    _make_dir(out_dir)
    torch.save(generator.state_dict(), "{}/G_epoch_{}.pth".format(out_dir, epoch))
    torch.save(
        discriminator.state_dict(), "{}/D_epoch_{}.pth".format(out_dir, epoch)
    )

def _save_images(fig, epoch, n_batch, comment=""):
    """
    save images from a figure.

    Parameters
    ----------
    fig : plt.figure
        the figure to save.
    epoch : int
        what epoch you are on.
    n_batch : int
        what batch are you on in the epoch.
    comment : string, optional
        part of the path to save. The default is "".

    Returns
    -------
    None.

    """
    out_dir = "./data/images/{}".format(data_subdir)
    _make_dir(out_dir)
    fig.savefig(
        "{}/{}_epoch_{}_batch_{}.png".format(out_dir, comment, epoch, n_batch)
    )


def save_torch_images(
    horizontal_grid, grid, epoch, n_batch, plot_horizontal=True
):
    out_dir = "./data/images/{}".format(data_subdir)
    _make_dir(out_dir)

    # Plot and save horizontal
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
    plt.axis("off")
    if plot_horizontal:
        display.display(plt.gcf())
    _save_images(fig, epoch, n_batch, "hori")
    plt.close()

    # Save squared
    fig = plt.figure()
    plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    plt.axis("off")
    _save_images(fig, epoch, n_batch)
    plt.close()



    # Private Functionality

   

def log_images(

    images,
    num_images,
    epoch,
    n_batch,
    num_batches,
    format="NCHW",
    normalize=True,
):
    """
    input images are expected in format (NCHW)
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    if format == "NHWC":
        images = images.transpose(1, 3)

    # Make horizontal grid from image tensor
    horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
    # Make vertical grid from image tensor
    nrows = int(np.sqrt(num_images))
    grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

    # Save plots
    save_torch_images(horizontal_grid, grid, epoch, n_batch)
 


# In[39]:


class DiscriminatorNet(torch.nn.Module):
    """A three hidden-layer discriminative neural network."""

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = int(np.prod(img_shape))
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024), nn.ELU(0.2), nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(0.2), nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256), nn.ELU(0.2), nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


discriminator = DiscriminatorNet()


def images_to_vectors(images):
    return images.view(images.size(0), int(np.prod(img_shape)))


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), nc, image_size, image_size)


# In[40]:


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = z_dim
        n_out = int(np.prod(img_shape))

        self.hidden0 = nn.Sequential(nn.Linear(n_features, 256), nn.ELU(0.2))
        self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.ELU(0.2))
        self.hidden2 = nn.Sequential(nn.Linear(512, 1024), nn.ELU(0.2))

        self.out = nn.Sequential(nn.Linear(1024, n_out), nn.Tanh())

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


generator = GeneratorNet()


def noise(size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    n = torch.randn(size, z_dim)
    return n


# In[41]:


d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
loss = nn.L1Loss()


# In[42]:


num_test_samples = 4
test_noise = noise(num_test_samples)


# In[ ]:


# this is to load models
def load_models(epoch):
    """
    Load the models saved at a given epoch.

    Parameters
    ----------
    epoch : int
        The epoch to load from.

    Returns
    -------
    None.

    """
    model_name = "wgan"
    data_name = "images"
    saved_dir = "./data/models/{}".format("{}/{}".format(model_name, data_name))
    discriminator.load_state_dict(
        torch.load("{}/D_epoch_{}.pth".format(saved_dir, epoch))
    )
    generator.load_state_dict(torch.load("{}/G_epoch_{}.pth".format(saved_dir, epoch)))


# In[19]:


# actually load
if load_from is not None:

    load_models(load_from)
else:
    load_from = 0


# test model on noise
def gen_image():
    """
    Genertate an image uding the model, and save as skin.png.

    Returns
    -------
    None.

    """
    plt.figure()
    # see what size px = 1/plt.rcParams['figure.dpi'] gives,
    # then solve 64x/plt.rcParams['figure.dpi']=size
    # to get 64 by 64 image set px=1/x
    # fro my screen this is 54
    # px = 1/plt.rcParams['figure.dpi']
    px = 1 / 54
    test_images = vectors_to_images(generator(noise(1)))
    test_images = test_images.data
    plt.figure(figsize=(image_size * px, image_size * px))
    plt.axis("off")
    plt.imshow(np.moveaxis(vutils.make_grid(test_images).numpy(), 0, -1))
    if transparency:
        plt.savefig("skin.png", bbox_inches="tight", pad_inches=0, transparent=True)
    else:
        plt.savefig("skin.png", bbox_inches="tight", pad_inches=0)


# In[ ]:

def show_image():
    """
    Genertate an image uding the model, and save as skin.png.

    Returns
    -------
    None.

    """
    plt.figure()
    test_images = vectors_to_images(generator(noise(1)))
    test_images = test_images.data
    plt.figure(figsize=(5, 5 ))
    plt.axis("off")
    plt.imshow(np.moveaxis(vutils.make_grid(test_images).numpy(), 0, -1))
for i in range(10):
    show_image()
