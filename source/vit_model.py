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
    
    '''
    Each patch is a square
    '''
    patch_size=h//no_of_patches_per_row
    
    '''
    The following code extracts patches from each image row wise and flattens the patches
    '''
    for idx,image in enumerate(images):
      for i in range(no_of_patches_per_row):
        for j in range(no_of_patches_per_row):
          
          patch=image[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
          patches[idx,i*no_of_patches_per_row+j]=patch.flatten()
          
    return patches
          
          
      
    
class ViT(nn.Module):
  def __init__(self, chw=(1, 28, 28), n_patches_per_row=7,hidden_dim=8):
    # Super constructor
    super(ViT, self).__init__()

    self.chw = chw # (C, H, W)
    self.n_patches_per_row = n_patches_per_row
    self.patch_dim=(chw[1]//n_patches_per_row,chw[2]//n_patches_per_row)
    self.patch_embedding_dim=chw[0]*self.patch_dim[0]*self.patch_dim[1]

    assert chw[1] % n_patches_per_row == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches_per_row == 0, "Input shape not entirely divisible by number of patches"
    
    '''
    1. Linear mapping.
    2. To reduce the dimensionality of embedding of a patch.
    '''
    self.linear_layer=nn.Linear(self.patch_embedding_dim,hidden_dim)
    
    
    '''
    Adding classification token at the beginning of patch sequence for each image in the batch
    '''
    self.cls_token=nn.Parameter(torch.ones(1,hidden_dim))

  
  
  def forward(self, images):
    
    '''
    Patchify the input images
    '''
    patches = patchify(images, self.n_patches_per_row)
    
    
    '''
    1.Map each patch's embedding to a lesser dimension. The size gets reduced to hidden_dim.
    2. This layer also has learnable weights.
    '''
    hidden_patches=self.linear_layer(patches)
    
    
    '''
    Adding cls token at the beggining of patch sequence
    '''
    cls_added_patch_sequence=torch.stack([torch.cat((self.cls_token,patch_sequence),dim=0) for patch_sequence in hidden_patches])
    
    
    
    
    
    return cls_added_patch_sequence
    
