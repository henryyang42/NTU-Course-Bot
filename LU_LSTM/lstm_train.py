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
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import gensim
import json
import tempfile
import re

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
ap.add_argument("out_model", type=str, help="")
ap.add_argument("out_vocab", type=str, help="word, label, intent vocabulary")
ap.add_argument("-i", "--epoch", type=int, default=10, help="# epochs")
ap.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="")
ap.add_argument("-c", "--cost", type=str, default="categorical_crossentropy", help="loss (cost) function")
ap.add_argument("-l", "--log", type=str, help="output prediction result for analysis")
ap.add_argument("-b", "--bi-direct", action="store_true", help="bidirectional LSTM")
ap.add_argument("-a", "--activation", default="relu", type=str, help="activation function")
args = ap.parse_args()

# prepare data
X = [] # list of sequences of indice
Y = [] # list of sequences of one-hot encoding of label e.g. [0 0 ... 0 1 0 ... 0]
Y2 = [] # list of intents
#TODO intent label

intent2idx = {}
idx2intent = []
word2idx = {"#": 0, "<UNK>":1}
idx2word = ["#", "<UNK>"] # "#" for padding
label2idx = {"#": 0}
idx2label = ["#"]
max_seq_len = 0
pat_split = re.compile(ur"\s+")
with codecs.open(args.sent_label_file, "r", "utf-8") as f_in:
    lines = f_in.readlines()
    print "# data:", len(lines)/3
    for i in range(0, len(lines), 3):
        # intent
        intent = lines[i].strip()
        if intent not in intent2idx:
            intent2idx[intent] = len(idx2intent)
            idx2intent.append(intent)
        Y2.append(intent2idx[intent])

        # tokens
        tokens = pat_split.split(lines[i+1].strip())
        #print tokens[0], repr(tokens[0])
        if len(tokens) > max_seq_len:
            max_seq_len = len(tokens)
        for t in tokens:
            #print t
            if t not in word2idx:
                word2idx[t] = len(idx2word)
                idx2word.append(t)
        idx_seq = seq_word2idx(tokens, word2idx)
        X.append(idx_seq)

        # BIO labels 
        labels = pat_split.split(lines[i+2].strip())
        if len(tokens) != len(labels):
            print "!! something wrong !!", len(tokens), len(labels)
            print " ".join(tokens)
            print " ".join(labels)
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(idx2label)
                idx2label.append(l)
        idx_seq = seq_word2idx(labels, label2idx)
        Y.append(idx_seq)

print "== reading data done =="
# pad sequences
X = sequence.pad_sequences(X, maxlen=max_seq_len)
Y = list(sequence.pad_sequences(Y, maxlen=max_seq_len))
print "== padding done =="

# convert BIO labels to one-hot encoding
#print Y.shape
#Y = np_utils.to_categorical(Y)
for i, y in enumerate(Y):
    """
    one_hot_seq = []
    for label_idx in y:
        vec = np.zeros((len(idx2label)))
        vec[label_idx] = 1.0
        one_hot_seq.append(vec)
    #print one_hot_seq, one_hot_seq.shape
    Y[i] = np.array(one_hot_seq)
    """
    Y[i] = np_utils.to_categorical(y, len(idx2label))

Y = np.array(Y)
print Y.shape

Y2 = np_utils.to_categorical(Y2)

##### contruct model #####

# [input layer]
seq_input = Input(shape=(max_seq_len,), dtype='int32')

# [embedding layer]
init_emb_W = None #TODO pre-trained word embedding?
embedding = Embedding(len(idx2word), args.emb_size, input_length=max_seq_len, dropout=0.2, weights=init_emb_W, trainable=True)(seq_input)

# [LSTM for slot]
if args.bi_direct:
    slot_lstm_out = Bidirectional(LSTM(args.emb_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))(embedding)
else:
    slot_lstm_out = LSTM(args.emb_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(embedding)

# [LSTM for intent]
intent_lstm_out = LSTM(args.emb_size, dropout_W=0.2, dropout_U=0.2)(slot_lstm_out)

# [transformation for slot]
x = TimeDistributed(Dense(args.emb_size))(slot_lstm_out)
#print x.__shape
x = Activation(args.activation)(x)
#print x.__shape
#TODO deeper feed-forward layers

# [output layer for slot]
x = TimeDistributed(Dense(len(idx2label)))(x)
#print x.__shape
slot_output = Activation('softmax', name='slot')(x)

# [output layer for intent]
x2 = Dense(len(idx2intent))(intent_lstm_out)
intent_output = Activation('softmax', name='intent')(x2)


# connect nodes to the model
model = Model(input=seq_input, output=[slot_output, intent_output])

##### ##### #####

cb = []
# early-stopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')
cb.append(earlyStopping)
# save best
best_weights_filepath = "/tmp/LU-LSTM_best_weights.hdf5"
print best_weights_filepath
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
cb.append(saveBestModel)

optm = Adam(lr=args.learning_rate)

intent_weight = 0.8
#TODO configurable loss weight
model.compile(loss=args.cost, optimizer=optm, loss_weights=[1.0-intent_weight, intent_weight])
print "== model compilation done =="

model.fit(X, [Y, Y2], validation_split=0.1, nb_epoch=args.epoch, callbacks=cb)


model.load_weights(best_weights_filepath)

# save model
model.save(args.out_model)

# dump_vocabulary
obj = {"word_vocab": idx2word, "intent_vocab": idx2intent, "slot_vocab": idx2label}
json.dump(obj, open(args.out_vocab, "w"))
