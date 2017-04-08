#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, merge
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.utils import np_utils

import pydot
from keras.utils.visualize_util import plot


ap = argparse.ArgumentParser()
ap.add_argument("model", type=str, help="Keras model to load")
ap.add_argument("out_file", type=str, help="output image")
args = ap.parse_args()

model = load_model(args.model)

#figure = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#display(figure)
#plot_model(model, to_file=args.out_file)
plot(model, to_file=args.out_file, show_shapes=True)
