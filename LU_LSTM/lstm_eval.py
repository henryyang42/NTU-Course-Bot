#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, merge
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.utils import np_utils
import json
import re
from LSTM_util import *

def eval_intent(true_intent_list, pred_intent_list):
    stat = {}

    n_data = len(true_intent_list)
    ## accuracy ##
    acc = 0.0
    for i in range(0, n_data):
        if true_intent_list[i] == pred_intent_list[i]:
            acc += 1
    acc /= n_data
    stat["accuracy"] = acc

    return stat

def eval_slot(true_labels_list, pred_labels_list):
    stat = {}

    n_data = len(true_labels_list)
    ## precision & recall ##
    TP = {}
    FP = {}
    FN = {}
    for i in range(0, n_data):
        true_labels = true_labels_list[i]
        pred_labels = pred_labels_list[i]
        seq_len = len(true_labels)
        for j in range(0, seq_len):
            #if "B" in true_labels[j] or "I" in true_labels[j]:
            if "B" in true_labels[j]:
                slot = true_labels[j][2:]
                if slot not in TP:
                    TP[slot] = 0.0
                    FP[slot] = 0.0
                    FN[slot] = 0.0
                if pred_labels[j] == true_labels[j]:
                    TP[slot] += 1
                else:
                    FN[slot] += 1
            #elif "B" in pred_labels[j] or "I" in pred_labels[j]:
            elif "B" in pred_labels[j]:
                slot = pred_labels[j][2:]
                if slot not in TP:
                    TP[slot] = 0.0
                    FP[slot] = 0.0
                    FN[slot] = 0.0
                FP[slot] += 1

    stat["precision"] = {}
    stat["recall"] = {}
    for slot in TP:
        if TP[slot] + FP[slot] == 0 or TP[slot] + FN[slot] == 0:
            continue
        slot_P = TP[slot] / (TP[slot] + FP[slot])
        slot_R = TP[slot] / (TP[slot] + FN[slot])
        stat["precision"][slot] = slot_P
        stat["recall"][slot] = slot_R

    return stat

#arguments
ap = argparse.ArgumentParser()
ap.add_argument("test_dataset", help="segmented test dataset, with intent & label")
ap.add_argument("model", type=str, help="Keras model to load")
ap.add_argument("vocab", type=str, help="idx2word table for word, slot, intent (in JSON format)")
args = ap.parse_args()

# load vocab
obj = json.load(open(args.vocab, "r"))
idx2label = obj["slot_vocab"]
idx2intent = obj["intent_vocab"]
word2idx = {}
for i, w in enumerate(obj["word_vocab"]):
    word2idx[w] = i

# load model
model = load_model(args.model)
seq_len =  model.input_layers[0].batch_input_shape[1]
#print "== load model done =="

# prediction on test data
with codecs.open(args.test_dataset, "r", "utf-8") as f_test:
    lines = f_test.readlines()
    seq_list = []
    len_list = []
    tokens_list = []
    true_intent_list = []
    true_labels_list = []
    n_data = len(lines) / 3
    for i in range(0, len(lines), 3):
        intent = lines[i].strip()
        tokens = lines[i+1].strip().split(" ")
        labels = lines[i+2].strip().split(" ")
        
        tokens_list.append(tokens)

        true_intent_list.append(intent)
        true_labels_list.append(labels)
    
        # pad sequence
        idx_seq = seq_word2idx(tokens, word2idx)
        if len(idx_seq) < seq_len:
            pad_idx_seq = [0]*(seq_len-len(idx_seq)) + idx_seq
        elif len(idx_seq) > seq_len:
            pad_idx_seq = idx_seq[-seq_len : ]
        else:
            pad_idx_seq = idx_seq
        len_list.append(len(idx_seq))
        seq_list.append(pad_idx_seq)

# predict
pred_label_vec_list, pred_intent_vec_list = model.predict(np.array(seq_list))

# convert result
pred_intent_list = []
pred_labels_list = []
for i in range(0, n_data):
    intent_idx = pred_intent_vec_list[i].argmax()
    pred_intent = idx2intent[intent_idx]
    pred_intent_list.append(pred_intent)

    slot_idx_seq = pred_label_vec_list[i].argmax(axis=-1)
    pred_labels = seq_idx2word(slot_idx_seq[-len_list[i] : ], idx2label)
    for j, l in enumerate(pred_labels):
        if l == '#':
            pred_labels[j] = 'O'
    if len(pred_labels) != len(true_labels_list[i]):
        print len(pred_labels), len(true_labels_list[i])

    pred_labels_list.append(pred_labels)

    # show error
    if pred_intent != true_intent_list[i]:
        print " ".join(tokens_list[i])
        print "Pred:", pred_intent, pred_labels
        print "True:", true_intent_list[i], true_labels_list[i]
    

intent_stat = eval_intent(true_intent_list, pred_intent_list)
print ("[intent] Accuracy:", intent_stat["accuracy"])

slot_stat = eval_slot(true_labels_list, pred_labels_list)
#print (slot_stat)
for slot in slot_stat["precision"]:
    print ("[slot - %s] Precision:" % slot, slot_stat["precision"][slot])
    print ("[slot - %s] Recall:" % slot, slot_stat["recall"][slot])
