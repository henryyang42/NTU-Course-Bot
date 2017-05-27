#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, LSTM
from keras.layers.core import Dropout, Flatten, RepeatVector, Permute
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, merge, multiply
from keras.layers.pooling import AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K
import gensim
import json
import tempfile
import re
from LSTM_util import *

#arguments
ap = argparse.ArgumentParser()

ap.add_argument("sent_label_file", help="example sentences(questions) with BIO slot labels")
ap.add_argument("emb_size", type=int, help="embedding size")
ap.add_argument("out_model", type=str, help="")
ap.add_argument("out_vocab", type=str, help="word, label, intent vocabulary")

ap.add_argument("-i", "--epoch", type=int, default=10, help="# epochs")
ap.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="")
ap.add_argument("-dr", "--dropout", type=float, default=0.2, help="")
ap.add_argument("-c", "--cost", type=str, default="categorical_crossentropy", help="loss (cost) function")

ap.add_argument("-b", "--bi-direct", action="store_true", help="bidirectional LSTM")
ap.add_argument("-n", "--attention", action="store_true", help="use attention")
ap.add_argument("-a", "--activation", default="relu", type=str, help="activation function")
ap.add_argument("-iw", "--intent-weight", type=float, default=0.8, help="weight of the loss for intent")
ap.add_argument("-rr", "--recur-reg", type=float, default=None, help="recurrent layer regularizer")
ap.add_argument("-bn", "--batch-norm", action="store_true", help="use BatchNormalization layer between LSTM")
ap.add_argument("-bal", "--balanced", action="store_true", help="balance class weights")

ap.add_argument("-we", "--word-emb", type=str, default=None, help="CWE word embedding")
ap.add_argument("-ce", "--char-emb", type=str, default=None, help="CWE character embedding")

args = ap.parse_args()

# prepare data
X = [] # list of sequences of indice
Y = [] # list of sequences of one-hot encoding of label e.g. [0 0 ... 0 1 0 ... 0]
Y2 = [] # list of intents

intent2idx = {}
idx2intent = []
word2idx = {"#": 0, "<UNK>":1}
idx2word = ["#", "<UNK>"] # "#" for padding
label2idx = {"#": 0}
idx2label = ["#"]
max_seq_len = 0
pat_split = re.compile(r"\s+")
with codecs.open(args.sent_label_file, "r", "utf-8") as f_in:
    lines = f_in.readlines()
    n_data = len(lines)/3
    print ("# data:", n_data)
    for i in range(0, len(lines), 3): 
        # verify data
        intent = lines[i].strip()
        tokens = pat_split.split(lines[i+1].strip()) 
        labels = pat_split.split(lines[i+2].strip())
        if len(tokens) != len(labels):
            print ("[error] # tokens & # labels do not match", len(tokens), len(labels))
            print (" ".join(tokens))
            print (" ".join(labels))
            continue # skip instance

        # intent
        if intent not in intent2idx:
            intent2idx[intent] = len(idx2intent)
            idx2intent.append(intent)

        # tokens
        if len(tokens) > max_seq_len:
            max_seq_len = len(tokens)
        for t in tokens:
            #print t
            if t not in word2idx:
                word2idx[t] = len(idx2word)
                idx2word.append(t)
        x_idx_seq = seq_word2idx(tokens, word2idx)

        # BIO labels 
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(idx2label)
                idx2label.append(l)
        y_idx_seq = seq_word2idx(labels, label2idx)
        
        # valid data
        X.append(x_idx_seq)
        Y.append(y_idx_seq)
        Y2.append(intent2idx[intent])
print ("Vocab. size:", len(idx2word))
print ("== reading data done ==")
# pad sequences
X = sequence.pad_sequences(X, maxlen=max_seq_len)
Y = list(sequence.pad_sequences(Y, maxlen=max_seq_len))
print ("== padding done ==")

# compute class weights
if args.balanced:
    slot_cw = {}
    for y_idx_seq in Y:
        for idx in y_idx_seq:
            if idx not in slot_cw:
                slot_cw[idx] = 0
            slot_cw[idx] += 1
    # weight = N / cnt
    for idx in slot_cw:
        slot_cw[idx] = n_data / slot_cw[idx]
    
    intent_cw = {} 
    for idx in Y2:
        if idx not in intent_cw:
            intent_cw[idx] = 0
        intent_cw[idx] += 1 
    # weight = N / cnt
    for idx in intent_cw:
        intent_cw[idx] = n_data / intent_cw[idx]

    print (slot_cw)
    print (intent_cw)

# convert BIO labels to one-hot encoding
#print Y.shape
for i, y in enumerate(Y):
    Y[i] = np_utils.to_categorical(y, len(idx2label))

Y = np.array(Y)
print (Y.shape)

Y2 = np_utils.to_categorical(Y2)

##### regularizer #####
r_reg = None
if args.recur_reg is not None:
    r_reg = regularizers.l2(args.recur_reg)

##### contruct model #####

# [input layer]
seq_input = Input(shape=(max_seq_len,), dtype='int32')

# [embedding layer]
init_emb_W = None #TODO pre-trained word embedding?
embedding = Embedding(len(idx2word), args.emb_size, input_length=max_seq_len, weights=init_emb_W, trainable=True)(seq_input)
embedding = Dropout(args.dropout)(embedding)

# [LSTM for slot]
if args.bi_direct:
    slot_lstm_out = Bidirectional(LSTM(args.emb_size, dropout=args.dropout, recurrent_dropout=args.dropout, return_sequences=True), name='slot LSTM', recurrent_regularizer=r_reg)(embedding)
else:
    slot_lstm_out = LSTM(args.emb_size, dropout=args.dropout, recurrent_dropout=args.dropout, return_sequences=True, name='slot LSTM', recurrent_regularizer=r_reg)(embedding)

if args.batch_norm:
    slot_lstm_out = BatchNormalization()(slot_lstm_out)

# [LSTM for intent]
if args.attention:
    intent_lstm_out = LSTM(args.emb_size, dropout=args.dropout, recurrent_dropout=args.dropout, name='intent LSTM', return_sequences=True, recurrent_regularizer=r_reg)(slot_lstm_out)
    attn = TimeDistributed(Dense(1, activation=args.activation))(intent_lstm_out)
    attn = Flatten()(attn)
    attn = Activation('softmax')(attn)
    attn = RepeatVector(args.emb_size)(attn)
    attn = Permute([2, 1])(attn)
    
    #intent_lstm_out = merge([intent_lstm_out, attn], mode='mul')
    intent_lstm_out = multiply([intent_lstm_out, attn])
    intent_lstm_out = AveragePooling1D(max_seq_len)(intent_lstm_out)
    intent_lstm_out = Flatten()(intent_lstm_out)

else:
    intent_lstm_out = LSTM(args.emb_size, dropout=args.dropout, recurrent_dropout=args.dropout, name='intent LSTM')(slot_lstm_out)

# [transformation for slot]
x = TimeDistributed(Dense(args.emb_size), name='slot transformation 1')(slot_lstm_out)
x = Activation(args.activation)(x)
#TODO deeper feed-forward layers

# [output layer for slot]
x = TimeDistributed(Dense(len(idx2label)), name='slot transformation 2')(x)
slot_output = Activation('softmax', name='slot')(x)

# [output layer for intent]
x2 = Dense(len(idx2intent), name='intent transformation')(intent_lstm_out)
intent_output = Activation('softmax', name='intent')(x2)


# connect nodes to the model
model = Model(inputs=seq_input, outputs=[slot_output, intent_output])

##### ##### #####

cb = []
# early-stopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')
cb.append(earlyStopping)
# save best
best_weights_filepath = "/tmp/LU-LSTM_best_weights.hdf5"
print (best_weights_filepath)
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
cb.append(saveBestModel)

optm = Adam(lr=args.learning_rate)

intent_weight = args.intent_weight
model.compile(loss=args.cost, optimizer=optm, loss_weights=[1.0-intent_weight, intent_weight])
print ("== model compilation done ==")

if args.balanced:
    #model.fit(X, [Y, Y2], validation_split=0.1, epochs=args.epoch, callbacks=cb, class_weight=[slot_cw, intent_cw])
    model.fit(X, [Y, Y2], validation_split=0.1, epochs=args.epoch, callbacks=cb, class_weight=[None, intent_cw])
else:
    model.fit(X, [Y, Y2], validation_split=0.1, epochs=args.epoch, callbacks=cb)


model.load_weights(best_weights_filepath)

# save model
model.save(args.out_model)

# dump_vocabulary
obj = {"word_vocab": idx2word, "intent_vocab": idx2intent, "slot_vocab": idx2label}
json.dump(obj, open(args.out_vocab, "w"))
