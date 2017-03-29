#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, merge
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import gensim
import json
import tempfile

def seq_word2idx(tokens, word2idx):
    idx_seq = []
    for t in tokens:
        if t in word2idx:
            idx_seq.append(word2idx[t])
        else:
            idx_seq.append(word2idx["<UNK>"])
    return idx_seq


#arguments
ap = argparse.ArgumentParser()
ap.add_argument("sent_label_file", help="example sentences(questions) with BIO slot labels")
ap.add_argument("emb_size", type=int, help="embedding size")
ap.add_argument("-i", "--epoch", type=int, default=10, help="# epochs")
ap.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="")
ap.add_argument("-c", "--cost", type=str, default="binary_crossentropy", help="loss (cost) function")
ap.add_argument("-l", "--log", type=str, help="output prediction result for analysis")
ap.add_argument("-b", "--bi-direct", action="store_true", help="bidirectional LSTM")
ap.add_argument("-a", "--activation", default="relu", type=str, help="activation function")
args = ap.parse_args()

# prepare data
X = [] # list of sequences of indice
Y = [] # list of sequences of one-hot encoding of label e.g. [0 0 ... 0 1 0 ... 0]
#TODO intent label

word2idx = {"#": 0, "<UNK>":1}
idx2word = ["#", "<UNK>"] # "#" for padding
label2idx = {"#": 0}
idx2label = ["#"]
max_seq_len = 0
with codecs.open(args.sent_label_file, "r", "utf-8") as f_in:
    lines = f_in.readlines()
    for i in range(0, len(lines), 2):
        # tokens
        tokens = lines[i].strip().split(" ")
        if len(tokens) > max_seq_len:
            max_seq_len = len(tokens)
        for t in tokens:
            if t not in word2idx:
                word2idx[t] = len(idx2word)
                idx2word.append(t)
        idx_seq = seq_word2idx(tokens, word2idx)
        X.append(idx_seq)

        # BIO labels 
        labels = lines[i+1].strip().split(" ")
        if len(tokens) != len(labels):
            print "!! something wrong !!"
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(idx2label)
                idx2label.append(l)
        idx_seq = seq_word2idx(labels, label2idx)
        Y.append(idx_seq)

# pad sequences
X = sequence.pad_sequences(X, maxlen=max_seq_len)
Y = sequence.pad_sequences(Y, maxlen=max_seq_len)

# convert BIO labels to one-hot encoding
for i, y in enumerate(Y)
    one_hot_seq = []
    for label_idx in y:
        vec = np.zeros((len(idx2label)))
        vec[label_idx] = 1.0
        one_hot_seq.append(vec)
    Y[i] = one_hot_seq

# [input layer]
seq_input = Input(shape=(max_seq_len,), dtype='int32')

# [embedding layer]
init_emb_W = None #TODO pre-trained word embedding?
embedding = Embedding(len(idx2word), args.emb_size, input_length=max_seq_len, dropout=0.2, weights=init_emb_W, trainable=seq_cfg["trainable"])(seq_input)

# [LSTM for slot]
if args.bi_direct:
    slot_lstm_out = Bidirectional(LSTM(args.emb_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))(embedding)
else:
    slot_lstm_out = LSTM(args.emb_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(embedding)

# [LSTM for intent]
intent_lstm_out = LSTM(args.emb_size, dropout_W=0.2, dropout_U=0.2)(slot_lstm_out)

# [transformation for slot]
x = TimeDistributed(Dense(args.emb_size))(slot_lstm)
x = Activation(args.activation)(x)
#TODO deeper feed-forward layers

# [output layer for slot]
x = TimeDistributed(Dense(args.emb_size))(slot_lstm_out)
slot_output = Activation('softmax')(x)

# [output layer for intent]
x2 = Dense(args.emb_size)(intent_lstm_out)
intent_output = Activation('softmax')(x)

# connect nodes to the model
model = Model(input=seq_input, output=slot_output)#TODO ad intent output


#TODO save model
