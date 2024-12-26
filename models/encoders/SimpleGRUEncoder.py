import torch
import torch.nn as nn

class SimpleGRUEncoder(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, latent_dim, timestep, batch_first=False): 
        super(SimpleGRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=batch_first)
        self.projector = nn.Linear(self.hidden_dim*timestep, latent_dim)

        self.ts = timestep

    def forward(self, x):
        batch = len(x)
        input = x.reshape(batch, self.ts, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return self.projector(output.flatten(start_dim=1))