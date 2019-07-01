import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import gin
import numpy as np

from rl_learn.modules import MLP


@gin.configurable
class LEARN(nn.Module):
    def __init__(self, 
        vocab_size,
        n_actions,
        n_layers=2,
        emb_size=50,
        infersent_emb_size=4096,
        dropout=0.8,
        d1=128,
        d2=128
    ):
        super(LEARN, self).__init__()
        
        self.act_gru = nn.GRU(1, d1, num_layers=n_layers, batch_first=True)

        self.emb = nn.Embedding(vocab_size, emb_size)

        self.gru = nn.GRU(emb_size, d2, num_layers=n_layers, batch_first=True)
        self.concat_mlp = MLP(d1+d2, 2, dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, actions, act_lengths, langs, lang_lengths):
        packed_acts = pack_padded_sequence(actions, act_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        act_out, _ = self.act_gru(packed_acts)
        act_out = torch.mean(act_out, 1)

        langs = langs.long()
        langs = self.emb(langs)

        packed_langs = pack_padded_sequence(langs, langlengths.cpu().numpy(), batch_first=True, enforce_sorted=False)

        packed_langs = packed_langs.float()
        packed_out, _ = self.gru(packed_langs)
        text_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        text_out = torch.mean(text_out, 1)

        out = torch.cat((text_out, act_out), 1)

        return self.concat_mlp(out)                                                                                                                                                                                                                               