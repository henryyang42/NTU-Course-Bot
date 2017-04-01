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

#arguments
ap = argparse.ArgumentParser()
ap.add_argument("test_file", help="sentences(questions) to label slot and intent")
ap.add_argument("model", type=str, help="Keras model to load")
ap.add_argument("vocab", type=str, help="idx2word table for word, slot, intent (in JSON format)")
args = ap.parse_args()

# load vocab
obj = json.load(open(args.vocab, "r"))
idx2label = obj["slot_vocab"]
intent2label = obj["intent_vocab"]
word2idx = {}
for i, w in enumerate(obj["word_vocab"]):
    word2idx[w] = i

# load model
model = load_model(args.model)
print "== load model done =="

# prediction on test data
with codecs.open(args.test_fild, "r", "utf-8") as f_test:
    for line in f_test:
        tokens = line.strip().split(" ")
        idx_seq = seq_word2idx(tokens, word2idx)

        pred_slot, pred_intent = model.predict([idx_seq])
