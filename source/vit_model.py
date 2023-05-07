import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

'''
1. Setting random seed ensures reproducible results. When ever this code runs, same sequence of random initializations will be used.
2. Helpful for testing and debugging purposes.
'''
np.random.seed(0)
torch.manual_seed(0)

# Importing necessary classes
from mha import muliheaded_attention


def patchify(images, no_of_patches_per_row):
    
    n,c,h,w=images.shape
    
    assert h==w,"This method expects square images only"
    
    
    patches=torch.zeros(n,no_of_patches_per_row**2,c*h*w//(no_of_patches_per_row**2))
    
    print(patches.shape)
    





class MyViT(nn.Module):
  def __init__(self, chw=(1, 28, 28), n_patches=7):
    # Super constructor
    super(MyViT, self).__init__()

    # Attributes
    self.chw = chw # (C, H, W)
    self.n_patches = n_patches

    assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

  def forward(self, images):
    patches = patchify(images, self.n_patches)
    return patches
    
