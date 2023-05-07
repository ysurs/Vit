import torch
import torch.nn as nn

class muliheaded_attention(nn.Module):
    
    def __init__(self, hidden_state, no_of_heads):
        
        super(muliheaded_attention, self).__init__()
        self.hidden_state = hidden_state
        self.no_of_heads = no_of_heads
        self.softmax=nn.Softmax(dim=-1)
        
        assert hidden_state % no_of_heads==0, "Number of heads should completely divide hidden state dimension"
        
        '''
        1. The following is the dimension which each head will attend to.
        2. type casting because we need an integer value i.e dimensions attented by each head.
        '''
        dimensions_per_head = int(hidden_state/no_of_heads)
        self.dimensions_per_head=dimensions_per_head
        
        '''
        Each head will have its own set of q,k,v mappings
        '''
        self.q_mapping=nn.ModuleList([nn.Linear(dimensions_per_head, dimensions_per_head) for head in range(no_of_heads)])
        self.k_mapping=nn.ModuleList([nn.Linear(dimensions_per_head, dimensions_per_head) for head in range(no_of_heads)])
        self.v_mapping=nn.ModuleList([nn.Linear(dimensions_per_head, dimensions_per_head) for head in range(no_of_heads)])
        
        
        
        
        
    def forward(self, patch_sequences):
        
        '''
        1. patch_sequences is of shape (batch_size (no of images in a batch), no_of_patches per image, embedding_size of each patch)
        '''
        
        '''
        The following list contains the concatenated filtered attention for all images in the batch.
        '''
        after_mha=[]
        
        for patch_sequence in patch_sequences:
            
            print(patch_sequence.shape)
            filtered_attention_matrices=[]
            
            for h in range(self.no_of_heads):
                
                '''
                1. Each head having its own q,k,v values
                '''
                per_head_q=self.q_mapping[h]
                per_head_k=self.k_mapping[h]
                per_head_v=self.v_mapping[h]
                
                
                '''
                Each head will attend to a subsequence of the total embedding.
                '''
                sub_sequence_per_head=patch_sequence[:,h*(self.dimensions_per_head):(h+1)*(self.dimensions_per_head)]
                
                
                q=per_head_q(sub_sequence_per_head)
                k=per_head_k(sub_sequence_per_head)
                v=per_head_v(sub_sequence_per_head)
                
                
                '''
                1. The attention values obtained below tell us how important is one patch to all the other patches including itself. We do this for all patches.
                2. This matrix is a square matrix of the attention values.
                '''
                attention_per_head=self.softmax((q@k.T)/self.dimensions_per_head**0.5)
                
                
                '''
                We use attention matrix to find filtered attention values. A very good intuitive example is found here: https://youtu.be/mMa2PmYJlCo?t=775
                '''
                filtered_attention=attention_per_head@v
                
                
                '''
                We append the filtered attention obtained from each head to a list. After collecting these filtered_attention matrices, we will need to concatenate them horizontally.
                '''
                filtered_attention_matrices.append(filtered_attention)
                print(filtered_attention)
            
            after_mha.append(torch.hstack(filtered_attention_matrices))
            print(after_mha)