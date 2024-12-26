import torch
import torch.nn as nn

class SimpleProjection(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleProjection, self).__init__()
        self.projector = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.projector(x)
