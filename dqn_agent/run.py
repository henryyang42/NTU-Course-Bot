import json
import copy
import os
import pickle
import random
import sys
import numpy as np
from agent_dqn import AgentDQN

from util import *
sys.path.append(os.getcwd())
sys.path.append(os.path.pardir)
from misc_scripts.access_django import *
from utils.lu import multi_turn_lu3
from utils.nlg import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *
from utils.query import *
from dialog_system import DialogManager
from dqn_agent import dialog_config

"""
Launch a dialog simulation per the comm dand line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""


# if __name__ == "__main__":
all_courses = list(query_course({}).values())
np.random.shuffle(all_courses)
course_dict = {k: v for k, v in enumerate(all_courses)}
act_set = text_to_dict("./dqn_agent/dia_acts.txt")
slot_set = text_to_dict("./dqn_agent/slot_set.txt")
print("----------Data Pre-processing Done----------\n")

##########################################################################
#   @params run_mode: (type: int)
#       0   for display mode (NL)
#       1   for debug mode(Dia_Act)
#       2   for debug mode(Dia_Act and NL)
#       >=3 for no display(i.e. training)
#   @params auto_suggest: (type: int)
#       0   for no auto_suggest
#       1   for auto_suggest
##########################################################################
dialog_config.run_mode = 3
dialog_config.auto_suggest = 1
print("----------Dialog Configurations Setup Done----------\n")

##########################################################################
# Parameters for Agent (Deep-Q-Network Agent)
#   @params agent_params: parameters of agent (type: dict)
#   @params act_level: (type: int)
#       0   for user simulator is Dia_Act level
#       1   for user simulator is NL level
#   @params predict_mode: predict model for DQN (type: bool)
#   @params warm_start: (type: int)
#       use rule policy to fill the experience-replay pool at the beginning
#       0   no warm start
#       1   warm start for training
#   @params cmd_input_mode: (type: int)
#       0   for NL input
#       1   for Dia_Act input (this parameter is for AgentCmd only)
##########################################################################
agent_params = {}
agent_params['max_turn'] = 20
agent_params['epsilon'] = 0.01
agent_params['agent_run_mode'] = 3
agent_params['agent_act_level'] = 1
agent_params['experience_replay_pool_size'] = 1000
agent_params['dqn_hidden_size'] = 60
agent_params['batch_size'] = 16
agent_params['gamma'] = 0.9
agent_params['predict_mode'] = False
agent_params['trained_model_path'] = None
agent_params['warm_start'] = 1
agent_params['cmd_input_mode'] = 0
agent = AgentDQN(course_dict, act_set, slot_set, agent_params)
print("----------AgentDQN Setup Done----------\n")

##########################################################################
# Parameters for User Simulators
#   @params usersim_params: parameters of user simulator (type: dict)
#   @params slot_err_prob: slot level error probability (type: float)
#   @params slot_err_mode: which kind of slot err mode (type: int)
#       0   for slot_val only
#       1   for three errs
#   @params intent_err_prob: intent level error probability (type: float)
#   @params learning_phase: train/test/all, default is all. (type: str)
#                           The user goal set could be split into train and
#                           test set, or do not split (all). Here exists
#                           some randomness at the first sampled user action,
#                           even for the same user goal, the generated
#                           dialogue might be different
#       'all'     train + test
#       'train'   train only
#       'test'    test only
##########################################################################
usersim_params = {}
usersim_params['max_turn'] = 20
usersim_params['slot_err_probability'] = 0.
usersim_params['slot_err_mode'] = 0.
usersim_params['intent_err_probability'] = 0.
usersim_params['simulator_run_mode'] = 3
usersim_params['simulator_act_level'] = 1
usersim_params['learning_phase'] = 'train'
user_sim = RuleSimulator(all_courses)
# user_sim = RuleSimulator(course_dict, act_set, slot_set, usersim_params)
print("----------RuleSimulator Setup Done----------\n")

##########################################################################
# load trained NLG model (need to be transformed)
##########################################################################
# nlg_model_path = params['nlg_model_path']
# diaact_nl_pairs = params['diaact_nl_pairs']
# nlg_model = nlg()
# nlg_model.load_nlg_model(nlg_model_path)
# nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

# agent.set_nlg_model(nlg_model)
# user_sim.set_nlg_model(nlg_model)

##########################################################################
# load trained NLU model (need to be transformed)
##########################################################################
# nlu_model_path = params['nlu_model_path']
# nlu_model = nlu()
# nlu_model.load_nlu_model(nlu_model_path)

# agent.set_nlu_model(nlu_model)
# user_sim.set_nlu_model(nlu_model)

##########################################################################
# Dialog Manager
##########################################################################
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, course_dict, all_courses)
print("----------DialogManager Setup Done----------\n")

##########################################################################
#   Run num_episodes Conversation Simulations
##########################################################################
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}
simulation_epoch_size = 20
batch_size = 20
warm_start = 1
warm_start_epochs = 100
success_rate_threshold = 0.30
save_check_point = 10
agt = 9
params = {}
params['write_model_dir'] = './dqn_agent/models/'
params['trained_model_path'] = None
print("----------Parameters Setup Done----------\n")

""" Initialization of Best Model and Performance Records """
best_model = {}
best_res = {'avg_reward': float('-inf'), 'epoch': 0,
            'avg_turns': float('inf'),   'success_rate': 0}
best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0
performance_records = {}
performance_records['success_rate'] = {}
performance_records['avg_turns'] = {}
performance_records['avg_reward'] = {}
print("----------Performance Records Setup Done----------\n")


""" Save Model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.p' % (
        agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    # if agt == 9:
    checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    checkpoint['params'] = params
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print('saved model in %s' % (filepath, ))
    except Exception as e:
        print('Error: Writing model fails: %s' % filepath)
        print(e)


""" Save Performance Numbers """
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"))
        print('saved model in %s' % (filepath, ))
    except Exception as e:
        print('Error: Writing model fails: %s' % filepath)
        print(e)


""" Run N-Simulation Dialogues """
def simulation_epoch(simulation_epoch_size):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in range(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn()
            reward = user_sim.reward
            # cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode %s: Success" % (episode))
                else:
                    print("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

        cumulative_reward += user_sim.reward

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['avg_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['avg_turns'] = float(cumulative_turns) / simulation_epoch_size
    print("simulation success rate %s, avg reward %s, avg turns %s" %
          (res['success_rate'], res['avg_reward'], res['avg_turns']))
    return(res)


""" Warm_Start Simulation (by Rule Policy) """
def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in range(warm_start_epochs):
        dialog_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn()
            reward = user_sim.reward
            # cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("warm_start simulation episode %s: Success" %
                          (episode))
                else:
                    print("warm_start simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break

        cumulative_reward += user_sim.reward

    agent.warm_start = 2  # just a counter to avoid executing warm simulation twice
    res['success_rate'] = float(successes) / simulation_epoch_size
    res['avg_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['avg_turns'] = float(cumulative_turns) / simulation_epoch_size
    print("Warm_Start %s epochs, success rate %s, avg reward %s, avg turns %s" % (
        episode + 1, res['success_rate'], res['avg_reward'], res['avg_turns']))
    print("Current experience replay buffer size %s" %
          (len(agent.experience_replay_pool)))


""" Run Episodes """
def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    # if agt == 9 and params['trained_model_path'] == None and warm_start == 1:
    if warm_start == 1:
        print('warm_start starting ...\n')
        warm_start_simulation()
        print('warm_start finished, start RL training ...\n')

    for episode in range(count):
        print("Episode: %s" % (episode))
        dialog_manager.initialize_episode()
        episode_over = False

        while not episode_over:
            episode_over, reward = dialog_manager.next_turn()
            reward = user_sim.reward
            # cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    print("Successful Dialog!")
                    successes += 1
                else:
                    print("Failed Dialog!")

                cumulative_turns += dialog_manager.state_tracker.turn_count

        cumulative_reward += user_sim.reward
    # simulation
    # if agt == 9 and params['trained_model_path'] == None:
        agent.predict_mode = True
        simulation_res = simulation_epoch(simulation_epoch_size)

        performance_records['success_rate'][episode] = simulation_res['success_rate']
        performance_records['avg_turns'][episode] = simulation_res['avg_turns']
        performance_records['avg_reward'][episode] = simulation_res['avg_reward']

        if simulation_res['success_rate'] >= best_res['success_rate']:
            if simulation_res['success_rate'] >= success_rate_threshold:  # threshold = 0.30
                agent.experience_replay_pool = []
                simulation_epoch(simulation_epoch_size)

        if simulation_res['success_rate'] > best_res['success_rate']:
            best_model['model'] = copy.deepcopy(agent)
            best_res['success_rate'] = simulation_res['success_rate']
            best_res['avg_reward'] = simulation_res['avg_reward']
            best_res['avg_turns'] = simulation_res['avg_turns']
            best_res['epoch'] = episode

        agent.clone_dqn = copy.deepcopy(agent.dqn)
        agent.train(batch_size, 1)
        agent.predict_mode = False

        print("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
            performance_records['success_rate'][episode], performance_records['avg_reward'][episode], performance_records['avg_turns'][episode], best_res['success_rate']))
        # save the model every 10 episodes
        if episode % save_check_point == 0 and params['trained_model_path'] == None:
            save_model(params['write_model_dir'], agt, best_res['success_rate'],
                       best_model['model'], best_res['epoch'], episode)
            save_performance_records(
                params['write_model_dir'], agt, performance_records)

        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (episode + 1, count,
                                                                                             successes, episode + 1, float(cumulative_reward) / (episode + 1), float(cumulative_turns) / (episode + 1)))
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (successes,
                                                                      count, float(cumulative_reward) / count, float(cumulative_turns) / count))
    status['successes'] += successes
    status['count'] += count

    # if agt == 9 and params['trained_model_path'] == None:
    save_model(params['write_model_dir'], agt, float(
        successes) / count, best_model['model'], best_res['epoch'], count)
    save_performance_records(
        params['write_model_dir'], agt, performance_records)


run_episodes(50, status)
