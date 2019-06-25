import torch
import torch.nn as nn
import torch.nn.functional as F

import gin


@gin.configurable
class MLP(nn.Module):
    def __init__(self, enc_size, output_size, dropout=1., n_layers=3, hidden_size=128):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        in_sizes = [enc_size] + [hidden_size] * (n_layers - 1)
        out_sizes = [hidden_size] * (n_layers - 1) + [output_size]
        self.linears = []
        for l in range(n_layers):
            self.linears.append(nn.Linear(in_sizes[l], out_sizes[l]))

        self.dropout = nn.Dropout(p=1-dropout)

        self.linears = nn.ModuleList(self.linears)

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            linear.bias.data.fill_(0.1)
    
    def forward(self, x):
        for l in range(self.n_layers - 1):
            x = self.linears[l](x)
            x = F.relu(x)
            with torch.no_grad():
                m = torch.mean(x, 0)
                v = torch.var(x, 0)
            x = F.batch_norm(x, m, v, training=self.training)
        
        return self.linears[-1](x)