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
        
        for patch_sequence in patch_sequences:
            
            print(patch_sequence.shape)
            
            for h in range(self.no_of_heads):
                
                '''
                1. Each head having its own q,k,v values
                '''
                per_head_q=self.q_mapping[h]
                per_head_k=self.k_mapping[h]
                per_head_v=self.v_mapping[h]
                
                
                sub_sequence_per_head=patch_sequence[:,h*(self.dimensions_per_head):(h+1)*(self.dimensions_per_head)]
                print(sub_sequence_per_head.shape)
                print(sub_sequence_per_head)
                