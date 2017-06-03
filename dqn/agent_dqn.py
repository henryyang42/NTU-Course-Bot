'''
Created on May 12, 2017

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>

@author: Haley
'''

import copy
import json
import os
import re
import pickle
import random
import sys
import numpy as np
from keras.models import load_model
sys.path.append(os.getcwd())
sys.path.append(os.path.pardir)

from dqn.util import *
from dqn.qlearning.utils import *
from dqn.dialog_config import *
from dqn.qlearning import DQN
from misc_scripts.access_django import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *
from utils.query import *
from utils.nlg import *
from DiaPol_rule.dia_pol import *



class AgentDQN():
    def __init__(self, course_dict=None, act_set=None, slot_set=None, params={}):
        self.course_dict = course_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.hidden_size = params.get('hidden_size', 50)
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 4

        # act_cardinality: 17 * 2 + 7 * 7 + 3 + 1 = 34 + 63 + 3 + 20 + 4 = 120
        self.state_dimension = 2 * self.act_cardinality + \
                               5 * self.slot_cardinality + \
                               1 + \
                               self.max_turn
        # print("Agent-DQN - __init__ -> state_dimension:\n\t", self.state_dimension, '\n')
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model (in our case, default = False)
        if params['trained_model_path'] != None:
            self.model = load_model(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2
        else:
            self.model = build_model(self.state_dimension, params['model_params'])

    def initialize_episode(self):
        """ Initialize a new episode.
            This function is called every time a new episode is run.
        """
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['title', 'instructor', 'classroom', 'schedule_str']
        # self.request_set = ['required_elective', 'sel_method', 'designated_for',
        #                     'schedule_str', 'classroom', 'instructor',
        #                     'title', 'serial_no']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation, state)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        user_action = state['user_action']
        current_slots = state['current_slots']
        agent_last = state['agent_action']

        ##################################################################
        #   Create one-hot of acts to represent the current user action
        ##################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ##################################################################
        #     Create bag of inform slots representation to represent the
        #     current user action
        ##################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ##################################################################
        #   Create bag of request slots representation to represent the
        #   current user action
        ##################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ##################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ##################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ##################################################################
        #   Encode last agent act
        ##################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ##################################################################
        #   Encode last agent inform slots
        ##################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ##################################################################
        #   Encode last agent request slots
        ##################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ##################################################################
        #  One-hot representation of the turn count?
        ##################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep,
                                               user_request_slots_rep, agent_act_rep,
                                               agent_inform_slots_rep, agent_request_slots_rep,
                                               current_slots_rep, turn_rep, turn_onehot_rep])
        return self.final_representation

    def run_policy(self, representation, state=None):
        """ epsilon-greedy policy """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy(state)
            else:
                return self.dqn.keras_predict(representation, self.model)


    def rule_policy(self, state=None):
        """ Rule Policy """
        ####################################################################
        #   Idiot Rule Policy
        ####################################################################
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['choice'] = []
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            self.phase += 1
            act_slot_response = {'diaact': "inform",
                                 'choice': [],
                                 'inform_slots': {},
                                 'request_slots': {}}
        elif self.phase == 1:
            act_slot_response = {'diaact': "closing",
                                 'choice': [],
                                 'inform_slots': {},
                                 'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("Action Index Not Found")
        return None

    def add_nl_to_action(self, agent_action, user_action, state):
        """ Add NL to Agent Dia_Act """

        # system_sentence = agent2nl(agent_action)
        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            agent_action['act_slot_response']['nl'] = agent2nl(
                self.refine_action(agent_action['act_slot_response'], state))
            # agent_action['act_slot_response']['nl'] = user_action['nl']
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            agent_action['act_slot_value_response']['nl'] = agent2nl(
                self.refine_action(agent_action['act_slot_value_response'], state))
            # agent_action['act_slot_response']['nl'] = user_action['nl']

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t,
                            state_tplus1_rep, episode_over)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            self.experience_replay_pool.append(training_example)

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """
        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            for iter in range(len(self.experience_replay_pool) // (batch_size)):
                batches = [random.choice(self.experience_replay_pool) for _ in range(batch_size)]
                # print("Agent-DQN - train -> shape(batch[0]):\n\t", np.shape(batches[0]), '\n')

                # batch_struct = {'cost': {'reg_cost': reg_cost, 'loss_cost': loss_cost, 'total_cost': loss_cost + reg_cost}
                #                 'grads': grads(float)}
                # batch_struct = self.dqn.singleBatch(batches, {'gamma': self.gamma}, self.clone_dqn)
                # print("Agent-DQN - train -> batch[0]:\n\t", batches[0], '\n')

                # self.cur_bellman_err += batch_struct['cost']['total_cost']

                self.cur_bellman_err += self.dqn.keras_train(batches, self.model)
                # print("Agent-DQN - train -> loss:\n\t", loss, '\n')

            print("Current Bellman Error: %.4f, Experience-Replay Pool Size: %s" %
                  (float(self.cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))


    @staticmethod
    def refine_action(act_slot_response, status):
        sys_action = get_action_from_frame(status['current_slots'])
        agent_action = copy.deepcopy(act_slot_response)
        # print("AgentDQN - refine_action: sys_action\n\t", sys_action, '\n')
        ####################################################################
        #   Handles the act_slot response (with values needing to be filled)
        ####################################################################
        inform_slots = sys_action.get('inform_slots', {})
        request_slots = sys_action.get('request_slots', {})
        choice_slots = sys_action.get('choice', [])

        if agent_action['diaact'] == 'multiple_choice':
            agent_action.update({
                'diaact': 'multiple_choice',
                'choice': choice_slots,
                'inform_slots': {},
                'request_slots': {},
                'turn': status['turn']})
        elif agent_action['diaact'] == 'inform':
            agent_action.update({
                'diaact': 'inform',
                'choice': [],
                'inform_slots': inform_slots,
                'request_slots': {},
                'turn': status['turn']})
        elif re.compile(r'request').search(agent_action['diaact']):
            agent_action.update({
                'diaact': 'request',
                'choice': [],
                'inform_slots': {},
                'request_slots': request_slots,
                'turn': status['turn']})
        elif agent_action['diaact'] == 'confirm':
            agent_action.update({
                'diaact': 'inform',
                'choice': [],
                'inform_slots': inform_slots,
                'request_slots': {},
                'turn': status['turn']})
        else:
            agent_action.update({
                'diaact': sys_action['diaact'],
                'choice': choice_slots,
                'inform_slots': inform_slots,
                'request_slots': request_slots,
                'turn': status['turn']})
        return agent_action
    ######################################################################
    #    Debug Functions
    ######################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path, ))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path, ))
            print(e)

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        print("trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return model



