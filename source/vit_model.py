import torch
import numpy as np
import torch.nn as nn

from mha import muliheaded_attention

'''
1. Setting random seed ensures reproducible results. When ever this code runs, same sequence of random initializations will be used.
2. Helpful for testing and debugging purposes.
'''
np.random.seed(0)
torch.manual_seed(0)

# Importing necessary classes
from mha import muliheaded_attention

'''
Patchifying images
'''
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
          

'''
1.For every patch, the values at i'th postion will be altered depending on i being odd or even.
2. For every patch in the sequence, a unique position is added.
'''
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
      
    


'''
This class implements ViT encoder which includes layer norm layers, mlp and multiheaded attention
'''
class ViT_encoder(nn.Module):
  
  def __init__(self,hidden_state,no_of_heads=2,multiplicative_factor=4):
    
    super(ViT_encoder,self).__init__()
    
    self.layer_norm1=nn.LayerNorm(hidden_state)
    self.layer_norm2=nn.LayerNorm(hidden_state)
    
    self.mha=muliheaded_attention(hidden_state,no_of_heads)
    
    self.mlp = nn.Sequential(
            nn.Linear(hidden_state, multiplicative_factor * hidden_state),
            nn.GELU(),
            nn.Linear(multiplicative_factor * hidden_state, hidden_state)
        )
    
    
  def forward(self,input):
    
    output1=input+self.mha(self.layer_norm1(input))
    output_from_vit_encoder=output1+self.mlp(self.layer_norm2(output1))
    
    return output_from_vit_encoder
      
    



class ViT(nn.Module):
  
  def __init__(self, chw=(1, 28, 28), n_patches_per_row=7,hidden_dim=8,no_of_vit_encoders=2,no_mha_heads=2,no_of_output_classes=10,device="cuda"):
    # Super constructor
    super(ViT, self).__init__()

    self.chw = chw # (C, H, W)
    self.n_patches_per_row = n_patches_per_row
    self.patch_dim=(chw[1]//n_patches_per_row,chw[2]//n_patches_per_row)
    self.patch_embedding_dim=chw[0]*self.patch_dim[0]*self.patch_dim[1]
    self.hidden_dim=hidden_dim
    self.vit_encoders=nn.ModuleList([ViT_encoder(hidden_dim,no_mha_heads) for encoder in range(no_of_vit_encoders)])
    self.no_of_output_classes=no_of_output_classes
    self.device=device
    
    '''
    Note: We could have used get_positional_embeddings function as is in the forward method but then it would have been part of computation and its parameters might have updated.
    We don't want the following tensor to change its value i.e we want it to be constant. Hence we have to use nn.Parameter and then set requires_grad=False.
    '''
    self.positional_embedding=nn.Parameter(get_positional_embeddings(self.n_patches_per_row**2+1, self.hidden_dim))
    self.positional_embedding.requires_grad=False

    assert chw[1] % n_patches_per_row == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches_per_row == 0, "Input shape not entirely divisible by number of patches"
    
    '''
    1. Linear mapping.
    2. To reduce the dimensionality of embedding of a patch.
    '''
    self.linear_layer=nn.Linear(self.patch_embedding_dim,self.hidden_dim)
    
    
    '''
    Adding classification token at the beginning of patch sequence for each image in the batch
    '''
    self.cls_token=nn.Parameter(torch.randn(1,hidden_dim))
    
    
    '''
    For final classification
    '''
    self.classification_mlp=nn.Sequential(
      nn.Linear(self.hidden_dim,self.no_of_output_classes),
      nn.Softmax(dim=-1)
    )

  
  
  def forward(self, images):
    
    
    images_per_batch,channels,height,weight=images.shape
    
    
    '''
    Patchify the input images
    '''
    patches = patchify(images, self.n_patches_per_row).to(device)
    
    
    '''
    1.Map each patch's embedding to a lesser dimension. The size gets reduced to hidden_dim.
    2. This layer also has learnable weights.
    '''
    hidden_patches=self.linear_layer(patches)
    
    
    '''
    Adding cls token at the beggining of patch sequence
    '''
    cls_added_patch_sequence=torch.stack([torch.cat((self.cls_token,patch_sequence),dim=0) for patch_sequence in hidden_patches])
    
    
    '''
    Note: We are applying positional embedding after adding CLS token. 
    CLS token will be the first token in patch sequence for each image
    '''
    batch_repeated_positional_embedding=self.positional_embedding.repeat(images_per_batch,1,1)
    patches_with_positional_embedding=cls_added_patch_sequence+batch_repeated_positional_embedding
    
    
    ''' 
    Passing the patch_sequence through vit encoders
    '''
    for vit_encoder in self.vit_encoders:
      out=vit_encoder(patches_with_positional_embedding)
    
    only_cls_token_per_batch=out[:,0,:]
    
    return self.classification_mlp(only_cls_token_per_batch)
    
