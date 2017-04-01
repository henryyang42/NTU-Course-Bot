import numpy as np

def seq_word2idx(tokens, word2idx):
    idx_seq = []
    for t in tokens:
        if t in word2idx:
            idx_seq.append(word2idx[t])
        else:
            idx_seq.append(word2idx["<UNK>"])
    return idx_seq

def seq_idx2word(idx_seq, idx2word):
    word_seq = []
    for idx in idx_seq:
        if idx >=0 and idx < len(idx2word):
            word_seq.append(idx2word[idx])
        else:
            word_seq.append("<UNK>")
    return word_seq
