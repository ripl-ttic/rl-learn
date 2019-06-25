import torch
import torch.nn as nn
import torch.nn.functional as F

import gin


@gin.configurable
class MLP(nn.Module):
    def __init__(self, enc_size, output_size, dropout=1., n_hidden=2, hidden_size=128):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        in_sizes = [enc_size] + [hidden_size] * n_hidden
        out_sizes = [hidden_size] * n_hidden + [output_size]
        self.linears = []
        for l in range(n_hidden + 1):
            self.linears.append(nn.Linear(in_sizes[l], out_sizes[l]))

        self.batch_norms = []
        for l in range(n_hidden):
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        self.dropout = nn.Dropout(p=1-dropout)

        self.linears = nn.ModuleList(self.linears)
        self.batch_norms = nn.ModuleList(self.batch_norms)

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            linear.bias.data.fill_(0.1)
    
    def forward(self, x):
        for l in range(self.n_hidden):
            x = self.linears[l](x)
            x = F.relu(x)
            x = self.batch_norms[l](x)
        
        return self.linears[-1](x)