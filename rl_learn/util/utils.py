import numpy as np
import gin
import torch
import torch.nn as nn


def pad_seq_feature(seq, length):
    seq = np.asarray(seq)
    if length < np.size(seq, 0):
            return seq[:length]
    dim = np.size(seq, 1)
    result = np.zeros((length, dim))
    result[0:seq.shape[0], :] = seq
    return result

def pad_seq_onehot(seq, length):
    seq = np.asarray(seq)
    if length < np.size(seq, 0):
            return seq[:length]
    result = np.zeros(length)
    result[0:seq.shape[0]] = seq
    return result

def get_batch_lang_lengths(lang_list, lang_enc):
    if lang_enc == 'onehot':
        langs = []
        lengths = []
        for i, l in enumerate(lang_list):
            lengths.append(len(l))
            langs.append(np.array(pad_seq_onehot(l, 20)))
        
        langs = np.array(langs)
        lengths = np.clip(np.array(lengths), 0, 20)
        return langs, lengths
    elif lang_enc == 'glove':
        langs = []
        lengths = []
        for i, l in enumerate(lang_list):
            lengths.append(len(l))
            langs.append(np.array(pad_seq_feature(l, 20)))
        
        langs = np.array(langs)
        lengths = np.clip(np.array(lengths), 0, 20)
        return langs, lengths
    elif lang_enc == 'infersent':
        return np.asarray(lang_list), np.array([])
    else:
        raise NotImplementedError

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias