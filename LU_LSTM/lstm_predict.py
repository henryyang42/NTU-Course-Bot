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
try:
    from .LSTM_util import *
except:
    from LSTM_util import *
"""
# http://stackoverflow.com/questions/40154320/replicating-models-in-keras-and-tensorflow-for-a-multi-threaded-setting
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)
"""


def get_intent_slot(model, tokens, word2idx, idx2label, idx2intent):
    # prepare sequence input
    seq_len =  model.input_layers[0].batch_input_shape[1]
    idx_seq = seq_word2idx(tokens, word2idx)
    if len(idx_seq) < seq_len:
        pad_idx_seq = [0]*(seq_len-len(idx_seq)) + idx_seq
    elif len(idx_seq) > seq_len:
        pad_idx_seq = idx_seq[-seq_len : ]
    else:
        pad_idx_seq = idx_seq

    # predict
    #with sess.graph.as_default():
    pred_slot, pred_intent = model.predict(np.array([pad_idx_seq]))
    #print pred_slot
    #print pred_intent

    # convert result
    intent_idx = pred_intent[0].argmax()
    intent = idx2intent[intent_idx]

    slot_idx_seq = pred_slot[0].argmax(axis=-1)
    #print slot_idx_seq
    labels = seq_idx2word(slot_idx_seq[-len(tokens) : ], idx2label)
    for i, l in enumerate(labels):
        if l == '#':
            labels[i] = 'O'

    return intent, tokens, labels

def get_intent_slot_prob(model, tokens, word2idx, idx2label, idx2intent):
    # prepare sequence input
    seq_len =  model.input_layers[0].batch_input_shape[1]
    idx_seq = seq_word2idx(tokens, word2idx)
    if len(idx_seq) < seq_len:
        pad_idx_seq = [0]*(seq_len-len(idx_seq)) + idx_seq
    elif len(idx_seq) > seq_len:
        pad_idx_seq = idx_seq[-seq_len : ]
    else:
        pad_idx_seq = idx_seq

    # predict
    #with sess.graph.as_default():
    pred_slot, pred_intent = model.predict(np.array([pad_idx_seq]))
    #print pred_slot
    #print pred_intent

    # convert result
    intent_prob = {}
    for idx, p in enumerate(pred_intent[0]):
        intent = idx2intent[idx]
        intent_prob[intent] = p

    label_prob_list = []
    for slot_vec in pred_slot[0][-len(tokens) : ]:
        label_prob = {}
        for idx, p in enumerate(slot_vec):
            label = idx2label[idx]
            label_prob[label] = p
        label_prob['O'] += label_prob['#']
        del label_prob['#']
        label_prob_list.append(label_prob)

    return intent_prob, tokens, label_prob_list


#arguments
ap = argparse.ArgumentParser()
ap.add_argument("test_file", help="sentences(questions) to label slot and intent")
ap.add_argument("model", type=str, help="Keras model to load")
ap.add_argument("vocab", type=str, help="idx2word table for word, slot, intent (in JSON format)")


if __name__ == '__main__':
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
    #print "== load model done =="

    # prediction on test data
    with codecs.open(args.test_file, "r", "utf-8") as f_test:
        for line in f_test:
            tokens = line.strip().split(" ")

            print ("== Single-turn ==")
            intent, tokens, labels = get_intent_slot(model, tokens, word2idx, idx2label, idx2intent)
            print (intent)
            print (" ".join(tokens))
            print (" ".join(labels))

            print ("== Output prob. for DST ==")
            intent_prob, tokens, label_prob_list = get_intent_slot_prob(model, tokens, word2idx, idx2label, idx2intent)
            print (intent_prob)
            print (" ".join(tokens))
            print (label_prob_list)
