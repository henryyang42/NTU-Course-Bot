"""
Created on May 11, 2017

@author: Haley
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


def plot_sr(res, x_steps, sample_size):
    rate = res['success_rate']
    x, y = [], []
    for k, v in rate.items():
        x += [int(k) + 1]
        y += [v]

    vx, vy = zip(*sorted(zip(x, y), key=lambda x: int(x[0])))
    vx = np.array(vx[:sample_size])
    vy = np.array(vy[:sample_size])
    x_axis = np.arange(0, max(vx) + 1, sample_size / x_steps)
    y_axis = np.arange(0, 1.3, 0.1)

    plt.figure(figsize=(20, 10))
    plt.xlim(0, max(vx))
    plt.ylim(0, 1.2)

    plt.fill_between(vx, vy - np.std(vy), vy + np.std(vy), color="#D1E8FF")
    plt.plot(vx, vy, color="#108EFF")

    plt.xlabel('Simulation Epoch', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)

    plt.xticks(x_axis, fontsize=14)
    plt.yticks(y_axis, fontsize=14)

    plt.savefig('./dqn/checkpoints/success_rate.png')
    plt.close()


def plot_ar(res, x_steps, y_steps, sample_size, max_reward):
    rate = res['avg_reward']
    x, y = [], []
    for k, v in rate.items():
        x += [int(k) + 1]
        y += [v]

    vx, vy = zip(*sorted(zip(x, y), key=lambda x: int(x[0])))
    vx = np.array(vx[:sample_size])
    vy = np.array(vy[:sample_size])
    x_axis = np.arange(0, max(vx) + 1, sample_size / x_steps)
    y_axis = np.arange(-max_reward, max_reward + (max_reward / y_steps), 2 * max_reward / y_steps)

    plt.figure(figsize=(20, 10))
    plt.xlim(0, max(vx))
    plt.ylim(-max_reward, max_reward)

    plt.fill_between(vx, vy - np.std(vy), vy + np.std(vy), color="#FFEDCA")
    plt.plot(vx, vy, color="#FFAB00")

    plt.xlabel('Simulation Epoch', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)

    plt.xticks(x_axis, fontsize=14)
    plt.yticks(y_axis, fontsize=14)

    plt.savefig('./dqn/checkpoints/avg_reward.png')
    plt.close()


def plot_at(res, x_steps, y_steps, sample_size, max_turn):
    rate = res['avg_turns']
    x, y = [], []
    for k, v in rate.items():
        x += [int(k) + 1]
        y += [v]

    vx, vy = zip(*sorted(zip(x, y), key=lambda x: int(x[0])))
    vx = np.array(vx[:sample_size])
    vy = np.array(vy[:sample_size])
    x_axis = np.arange(0, max(vx) + 1, sample_size / x_steps)
    y_axis = np.arange(0, max_turn + (max_turn / y_steps), max_turn / y_steps)

    plt.figure(figsize=(20, 10))
    plt.xlim(0, max(vx))
    plt.ylim(0, max_turn)

    plt.fill_between(vx, vy - np.std(vy), vy + np.std(vy), color="#CBE6CB")
    plt.plot(vx, vy, color="#008300")

    plt.xlabel('Simulation Epoch', fontsize=16)
    plt.ylabel('Average Turns', fontsize=16)

    plt.xticks(x_axis, fontsize=14)
    plt.yticks(y_axis, fontsize=14)

    plt.savefig('./dqn/checkpoints/avg_turns.png')
    plt.close()
