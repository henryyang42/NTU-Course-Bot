"""
Created on May 11, 2017

@author: haley
"""

import numpy as np


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
