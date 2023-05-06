import torch.nn as nn

class muliheaded_attention(nn.Module):
    
    def __init__(self, hidden_state, no_of_heads):
        
        super(muliheaded_attention, self).__init__()
        self.hidden_state = hidden_state
        self.no_of_heads = no_of_heads
        
        assert hidden_state % no_of_heads==0, "Number of heads should completely divide hidden state dimension"
        
        '''
        1. The following is the dimension which each head will attend to.
        2. type casting because we need an integer value i.e dimensions attented by each head.
        '''
        dimensions_per_head = int(hidden_state/no_of_heads)
        
        print(dimensions_per_head)
        
        
    def forward():
        pass