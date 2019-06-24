import numpy as np
import gin


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
        lengths = np.array(lengths)
        return langs, lengths
    elif lang_enc == 'glove':
        langs = []
        lengths = []
        for i, l in enumerate(lang_list):
            lengths.append(len(l))
            langs.append(np.array(pad_seq_feature(l, 20)))
        
        langs = np.array(langs)
        lengths = np.array(lengths)
        return langs, lengths
    elif lang_enc == 'infersent':
        return lang_list, []
    else:
        raise NotImplementedError
