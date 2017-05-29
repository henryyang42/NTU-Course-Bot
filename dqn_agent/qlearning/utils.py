'''
Created on Jun 18, 2016

@author: xiul
'''

import math
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.pardir)
import numpy as np
from dqn_agent.dialog_config import *
from utils.decorator import run_once
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, SpatialDropout2D
from keras.layers import Dense
from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop
from keras.initializers import TruncatedNormal
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model


ACTIONS = len(feasible_actions)  # number of valid actions
GAMMA = 0.99999


def initWeight(n, d):
    scale_factor = math.sqrt(float(6) / (n + d))
    # scale_factor = 0.1
    return (np.random.rand(n, d) * 2 - 1) * scale_factor

""" for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
def mergeDicts(d0, d1):
    for k in d1:
        if k in d0:
            d0[k] += d1[k]
        else:
            d0[k] = d1[k]


@run_once
def build_model(batch, params={}):
    ##########################################################################
    #   Each batch:
    #      batch[0] = state_t_rep       (type: <numpy.ndarray>, shape: (1, 124))
    #      batch[1] = action_index      (type: <int>)
    #      batch[2] = reward            (type: <int>)
    #      batch[3] = state_tplus1_rep  (type: <numpy.ndarray>, shape: (1, 124))
    #      batch[4] = episode_over      (type: <bool>)
    #   Format: (s_t, a_t, r, s_{t+1}, episode_over)
    #            [0], [1], [2], [3], [4]
    ##########################################################################
    s_t_rep = batch[0]

    act_id = np.zeros((1, 1))
    act_id[0, 0] = batch[1]

    reward = np.zeros((1, 1))
    reward[0, 0] = batch[2]

    s_t1_rep = batch[3]

    epi_over = np.zeros((1, 1))
    epi_over[0, 0] = 0 if batch[4] else 1

    f_vector = s_t_rep.reshape(s_t_rep.shape[1], )
    # f_vector = np.hstack([s_t_rep, act_id, reward, s_t1_rep, epi_over])

    lr = params.get('learning_rate', 0.001)
    momen = params.get('momentum', 0.1)
    grad_clip = params.get('grad_clip', -1e-3)
    smooth_eps = params.get('smooth_eps', 1e-8)
    opt = params.get('opt', 'adam')
    dp = params.get('dropout_rate', 0.1)
    activation_func = params.get('activation_func', 'relu')

    model_in = Input(shape=f_vector.shape, name='model_in')

    model_out = Dense(50, name='Dense_1')(model_in)

    model_out = Activation(activation_func, name='relu_1')(model_out)

    model_out = Dropout(dp, name='dp_1')(model_out)

    model_out = Dense(25, name='Dense_2')(model_out)

    model_out = Activation(activation_func, name='relu_2')(model_out)

    model_out = Dropout(dp, name='dp_2')(model_out)

    model_out = Dense(ACTIONS, name='Dense_3')(model_out)

    model_out = Activation('softmax', name='softmax_1')(model_out)

    model = Model(inputs=[model_in], outputs=[model_out])

    if opt == 'adadelta':
        # keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        optimizer = Adadelta(lr=float(lr))

    elif opt == 'adagrad':
        # keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        optimizer = Adagrad(lr=float(lr))

    elif opt == 'adam':
        # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        optimizer = Adam(lr=float(lr))

    elif opt == 'adamax':
        # keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        optimizer = Adamax(lr=float(lr))

    elif opt == 'nadam':
        # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        optimizer = Nadam(lr=float(lr))

    elif opt == 'rmsprop':
        # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = RMSprop(lr=float(lr))

    else:
        raise ValueError(("Optimizer Error! \'opt\' should be one of [\'adadelta\', \'adagrad\',\
                         \'adam\', \'adamax\', \'nadam\', \'rmsprop\']."))

    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model
