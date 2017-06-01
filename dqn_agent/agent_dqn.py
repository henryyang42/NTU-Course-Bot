'''
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python ./run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: xiul
'''

import copy
import json
import os
import pickle
import random
import sys
import numpy as np
from dialog_config import *
from qlearning import DQN

from util import *
from dqn_agent.qlearning.utils import *
sys.path.append(os.getcwd())
sys.path.append(os.path.pardir)
from misc_scripts.access_django import *
from utils.lu import multi_turn_lu3
from utils.nlg import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *
from utils.query import *
from DiaPol_rule.dia_pol import *

class AgentDQN():
    def __init__(self, course_dict=None, act_set=None, slot_set=None, params=None):
        self.course_dict = course_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

        self.experience_replay_pool_size = params.get(
            'experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 4

        # act_cardinality: 17 * 2 + 7 * 9 + 3 + 1 = 34 + 63 + 3 + 20 + 4 = 120
        self.state_dimension = 2 * self.act_cardinality + \
                               7 * self.slot_cardinality + \
                               3 + \
                               self.max_turn
        # print("Agent-DQN - __init__ -> state_dimension:\n\t", self.state_dimension, '\n')
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.cur_bellman_err = 0
        self.model = build_model(self.state_dimension, params['model_params'])

        # Prediction Mode: load trained DQN model (in our case, default = False)
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(
                self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode.
            This function is called every time a new episode is run.
        """
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['required_elective', 'sel_method', 'designated_for',
                            'schedule_str', 'classroom', 'instructor',
                            'title', 'serial_no']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation, state)
        # print("DQN-Agent - state_to_action -> self.action:\n\t", self.action, '\n')

        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        # print("DQN-Agent - state_to_action -> act_slot_response:\n\t", act_slot_response, '\n')

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ##################################################################
        #   Create one-hot of acts to represent the current user action
        ##################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
        # print("user_act_rep:\n\t", user_act_rep, '\n')

        ##################################################################
        #     Create bag of inform slots representation to represent the
        #     current user action
        ##################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0
        # print("user_inform_slots_rep:\n\t", user_inform_slots_rep, '\n')

        ##################################################################
        #   Create bag of request slots representation to represent the
        #   current user action
        ##################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0
        # print("user_request_slots_rep:\n\t", user_request_slots_rep, '\n')

        ##################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ##################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0
        # print("current_slots_rep:\n\t", current_slots_rep, '\n')

        ##################################################################
        #   Encode last agent act
        ##################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        # print("agent_act_rep:\n\t", agent_act_rep, '\n')

        ##################################################################
        #   Encode last agent inform slots
        ##################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0
        # print("agent_inform_slots_rep:\n\t", agent_inform_slots_rep, '\n')

        ##################################################################
        #   Encode last agent request slots
        ##################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.
        # print("agent_request_slots_rep:\n\t", agent_request_slots_rep, '\n')
        # print("turn_rep:", turn_rep)

        ##################################################################
        #  One-hot representation of the turn count?
        ##################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0
        # print("turn_onehot_rep:", turn_onehot_rep)

        ##################################################################
        #   Representation of KB results (scaled counts)
        ##################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + \
                        kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        # print("kb_count_rep:", kb_count_rep)

        ##################################################################
        #   Representation of KB results (binary)
        ##################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + \
            np.sum(kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(
                    kb_results_dict[slot] > 0.)
        # print("kb_binary_rep:", kb_binary_rep)

        self.final_representation = np.hstack([user_act_rep,user_inform_slots_rep,
                                               user_request_slots_rep, agent_act_rep,
                                               agent_inform_slots_rep, agent_request_slots_rep,
                                               current_slots_rep, turn_rep, turn_onehot_rep,
                                               kb_binary_rep, kb_count_rep])

        # print("Final Representation Dimension:", np.shape(self.final_representation))
        # print("final_representation:", self.final_representation)
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

        # sys_action = get_action_from_frame(state['current_slots'])
        # print("DQN-Agent - rule_policy -> sys_action (from get_action_from_frame)")
        # for k, v in sys_action.items():
        #     print('\t', "\"%s\":" % k, v)
        # print()

        ####################################################################
        #   Idiot Rule Policy
        ####################################################################
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['choice'] = []
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform",
                                'inform_slots': {},
                                'request_slots': {},
                                'choice': []}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "closing", 'inform_slots': {},
                                'request_slots': {}, 'choice': []}
        ####################################################################
        #   Old version follows kb_results
        ####################################################################
        # # if the unique course is found
        # if state['kb_results_dict']['matching_all_constraints'] == 1:
        #     act_slot_response = {}
        #     act_slot_response['diaact'] = "inform"
        #     act_slot_response['inform_slots'] = {
        #         list(state['current_slots']['request_slots'].keys())[0]: "PLACEHOLDER"}
        #     act_slot_response['request_slots'] = {}
        #     # print('DQN-Agent - rule_policy -> unique course is found: act_slot_response\n\t', act_slot_response, '\n')

        # # if there are multiple courses are found
        # elif state['kb_results_dict']['matching_all_constraints'] > 1:
        #     act_slot_response = {}
        #     act_slot_response['diaact'] = "multiple_choice"
        #     act_slot_response['choice'] = []
        #     act_slot_response['inform_slots'] = {}
        #     act_slot_response['request_slots'] = {}
        #     # print('DQN-Agent - rule_policy -> multiple courses are found: act_slot_response\n\t', act_slot_response, '\n')

        # elif self.phase == 0:
        #     act_slot_response = {'diaact': "inform",
        #                         'inform_slots': {},
        #                         'request_slots': {}}
        #     self.phase += 1
        # elif self.phase == 1:
        #     act_slot_response = {'diaact': "closing", 'inform_slots': {},
        #                          'request_slots': {}}
        ####################################################################
        #   New version follows get_action_from_frame
        ####################################################################
        # if sys_action['diaact'] == 'inform':
        #     act_slot_response = {}
        #     act_slot_response['diaact'] = "inform"
        #     act_slot_response['inform_slots'] = {
        #             k: "PLACEHOLDER" for k in sys_action['inform_slots'].keys()
        #         }
        #     act_slot_response['request_slots'] = {}
        #     # print('DQN-Agent - rule_policy -> unique course is found: act_slot_response\n\t', act_slot_response, '\n')

        # # if there are multiple courses are found
        # elif sys_action['diaact'] == 'multiple_choice':
        #     act_slot_response = {}
        #     act_slot_response['diaact'] = "multiple_choice"
        #     act_slot_response['choice'] = []
        #     act_slot_response['inform_slots'] = {}
        #     act_slot_response['request_slots'] = {}

        # # if the conditions of the course in not enough
        # else:
        #     # fill in the informed slot to agent's request set
        #     for slot in state['current_slots']['inform_slots'].keys():
        #         dict_slot = 'schedule_str' if slot == 'when' else slot
        #         self.request_set[dict_slot] = 1

        #     filled_in_slots_num = sum(list(self.request_set.values()))
        #     # print('DQN-Agent - rule_policy -> filled_in_slots_num\n\t', filled_in_slots_num, '\n')
        #     # print('DQN-Agent - rule_policy -> len(self.request_set)\n\t', len(self.request_set), '\n')

        #     # necessary slots not all filled in with correct values
        #     if filled_in_slots_num < len(self.request_set):
        #         slot = 'title'
        #         for k, v in self.request_set.items():
        #             if v == 0:
        #                 slot = k
        #                 break
        #         # print('DQN-Agent - rule_policy -> slot\n\t', slot, '\n')

        #         sys_act = get_action_from_frame(state['current_slots'])
        #         # print('DQN-Agent - rule_policy -> sys_act\n\t', sys_act, '\n')

        #         act_slot_response = {}
        #         act_slot_response['diaact'] = "request"
        #         act_slot_response['inform_slots'] = {}
        #         act_slot_response['request_slots'] = {slot: "UNK"}
        #         # print('DQN-Agent - rule_policy -> conditions of the course in not enough: act_slot_response\n\t', act_slot_response, '\n')

        #     elif self.phase == 0:
        #         act_slot_response = {'diaact': "inform",
        #                             'inform_slots': {},
        #                             'request_slots': {}}
        #         self.phase += 1
        #     elif self.phase == 1:
        #         act_slot_response = {'diaact': "closing", 'inform_slots': {},
        #                             'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("Action Index Not Found")
        return None

    def add_nl_to_action(self, agent_action, user_action):
        """ Add NL to Agent Dia_Act """

        # print('DQN-Agent - add_nl_to_action -> agent_action\n\t', agent_action, '\n')
        # print('DQN-Agent - add_nl_to_action -> user_action\n\t', user_action, '\n')

        # system_sentence = agent2nl(agent_action)
        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            # NLG
            agent_action['act_slot_response']['nl'] = user_action['nl']
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            # # NLG)
            agent_action['act_slot_response']['nl'] = user_action['nl']

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



