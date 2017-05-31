"""
Created on May 11, 2017

@author: haley
"""


import math
import json
import numpy as np
import matplotlib.pyplot as plt


##########################################################################
#   Some helper functions
##########################################################################
def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers)
        Input:
            @param path: path to the file
        Output:
            @param slot_set: dict of key-value pairs (k: text, v: line numbers)
    """
    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set

def unique_states(training_data):
    unique = []
    for datum in training_data:
        if contains(unique, datum[0]):
            pass
        else:
            unique.append(datum[0].copy())
    return unique

def contains(unique, candidate_state):
    for state in unique:
        if np.array_equal(state, candidate_state):
            return True
        else:
            pass
    return False


def plot_sr(res, step_size):
    rate = res['success_rate']
    x, y = [], []
    for k, v in rate.items():
        x += [int(k) + 1]
        y += [v]
    sx, sy = zip(*sorted(zip(x, y), key=lambda x: int(x[0])))
    sx = list(sx)
    x_axis = range(0, max(sx) + 1, step_size)
    y_axis = np.arange(0, 1.1, 0.1)

    plt.plot(sx, sy)

    plt.xlabel('Simulation Epoch')
    plt.ylabel('Success Rate')

    plt.xticks(x_axis)
    plt.yticks(y_axis)

    plt.savefig('./dqn_agent/checkpoints/success_rate.png')
    plt.close()


def plot_ar(res, x_step_size, y_step_size):
    rate = res['avg_reward']
    x, y = [], []
    for k, v in rate.items():
        x += [int(k) + 1]
        y += [v]
    sx, sy = zip(*sorted(zip(x, y), key=lambda x: int(x[0])))
    sx = list(sx)
    x_axis = range(0, max(sx) + 1, x_step_size)
    y_axis = np.arange(0, math.ceil(max(sy)) + y_step_size, y_step_size)

    plt.plot(sx, sy)

    plt.xlabel('Simulation Epoch')
    plt.ylabel('Average Reward')

    plt.xticks(x_axis)
    plt.yticks(y_axis)

    plt.savefig('./dqn_agent/checkpoints/avg_reward.png')
    plt.close()


def plot_at(res, step_size):
    rate = res['avg_turns']
    x, y = [], []
    for k, v in rate.items():
        x += [int(k) + 1]
        y += [v]
    sx, sy = zip(*sorted(zip(x, y), key=lambda x: int(x[0])))
    sx = list(sx)
    x_axis = range(0, max(sx) + 1, step_size)
    y_axis = np.arange(0, math.ceil(max(sy)), 0.25)

    plt.plot(sx, sy)

    plt.xlabel('Simulation Epoch')
    plt.ylabel('Average Turns')

    plt.xticks(x_axis)
    plt.yticks(y_axis)

    plt.savefig('./dqn_agent/checkpoints/avg_turns.png')
    plt.close()
