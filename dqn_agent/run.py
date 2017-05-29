import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dict_path', dest='dict_path', type=str,
    #                     default='./deep_dialog/data/dicts.v3.p', help='path to the .json dictionary file')
    # parser.add_argument('--movie_kb_path', dest='movie_kb_path', type=str,
    #                     default='./deep_dialog/data/movie_kb.1k.p', help='path to the movie kb .json file')
    parser.add_argument('--act_set', dest='act_set', type=str, default="./dqn_agent/dia_acts.txt",
                        help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default="./dqn_agent/slot_set.txt",
                        help='path to slot set; none for loading from labeled file')
    # parser.add_argument('--goal_file_path', dest='goal_file_path', type=str,
    #                     default='./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p', help='a list of user goals')
    # parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
    #                     default='./deep_dialog/data/dia_act_nl_pairs.v6.json', help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=20, type=int,
                        help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=50,
                        type=int, help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob',
                        default=0.00, type=float, help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0,
                        type=int, help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob',
                        default=0.00, type=float, help='the intent err probability')

    parser.add_argument('--agt', dest='agt', default=9, type=int,
                        help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
    parser.add_argument('--usr', dest='usr', default=0, type=int,
                        help='Select a user simulator. 0 is a Frozen user simulator.')

    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0,
                        help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

    # load NLG & NLU model
    # parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
    #                     default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p', help='path to model file')
    # parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
    #                     default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p', help='path to the NLU model file')

    parser.add_argument('--act_level', dest='act_level', type=int,
                        default=1, help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=3,
                        help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int,
                        default=1, help='0 for no auto_suggest; 1 for auto_suggest')
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode',
                        type=int, default=0, help='run_mode: 0 for NL; 1 for dia_act')

    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size',
                        type=int, default=1000, help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size',
                        type=int, default=50, help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=20, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode',
                        type=bool, default=False, help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size',
                        type=int, default=50, help='the size of validation set')
    parser.add_argument('--warm_start', dest='warm_start', type=int,
                        default=1, help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs',
                        type=int, default=100, help='the number of epochs for warm start')

    parser.add_argument('--trained_model_path', dest='trained_model_path',
                        type=str, default=None, help='the path for trained model')
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir',
                        type=str, default='./dqn_agent/checkpoints/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point',
                        type=int, default=10, help='number of epochs for saving model')

    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold',
                        type=float, default=0.5, help='the threshold for success rate')

    # parser.add_argument('--split_fold', dest='split_fold', default=5,
    #                     type=int, help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase',
                        default='train', type=str, help='train/test/all; default is all')

    args = parser.parse_args()
    params = vars(args)


all_courses = list(query_course({}).values())
np.random.shuffle(all_courses)
course_dict = {k: v for k, v in enumerate(all_courses)}
act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])
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
dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']
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
agt = params['agt']
agent_params = {}
agent_params['max_turn'] = params['max_turn']
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']
agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']
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
usr = params['usr']
usersim_params = {}
usersim_params['max_turn'] = params['max_turn']
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']
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
simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size']
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']
success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']
print("----------Parameters Setup Done----------\n")

""" Initialization of Best Model and Performance Records """
best_model = {}
best_res = {'avg_reward': float('-inf'), 'epoch': 0,
            'avg_turns': float('inf'),   'success_rate': 0}
# best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0
performance_records = {}
performance_records['success_rate'] = {}
performance_records['avg_turns'] = {}
performance_records['avg_reward'] = {}
print("----------Performance Records Setup Done----------\n")


""" Save Model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    if agt == 9:
        checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    checkpoint['params'] = params
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('save_model: Model saved in %s' % (filepath, ))
    except Exception as e:
        print('Error! save_model: Writing model fails: %s' % filepath)
        print('\t', e)

""" Save Keras Model """
def save_keras_model(path, agt, success_rate, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    checkpoint['params'] = params
    try:
        with open(filepath + '.p', 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        dialog_manager.agent.model.save(filepath + '.h5')
        print('save_keras_model: Model saved in %s' % (filepath, ))
    except Exception as e:
        print('Error! save_keras_model: Writing model fails: %s' % filepath)
        print('\t', e)

""" Save Performance Numbers """
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "w"))
        print('save_performance_records: Model saved in %s' % (filepath, ))
    except Exception as e:
        print('Error! save_performance_records: Writing model fails: %s' % filepath)
        print('\t', e)


""" Warm_Start Simulation (by Rule Policy) """
def warm_start_simulation(warm_start_epochs):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in range(warm_start_epochs):
        dialog_manager.initialize_episode()
        episode_over = False
        per_episode_reward = 0
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            per_episode_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("Warm_start Simulation Episode %s: Success" %
                          (episode))
                else:
                    print("Warm_start Simulation Episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break

        print("\twarm_start_simulation:", per_episode_reward)
        # cumulative_reward += user_sim.reward

    agent.warm_start = 2  # just a counter to avoid executing warm simulation twice
    res['success_rate'] = float(successes) / warm_start_epochs
    res['avg_reward'] = float(cumulative_reward) / warm_start_epochs
    res['avg_turns'] = float(cumulative_turns) / warm_start_epochs
    print("Func - \"warm_start_simulation\":\n\t%s Epochs\n\tSuccess Rate %s\n\tAvg Reward %s\n\tAvg Turns %s" % (
        episode + 1, res['success_rate'], res['avg_reward'], res['avg_turns']))
    print("Current Experience-Replay Buffer Size %s" %
          (len(agent.experience_replay_pool)), '\n')


""" Run N-Simulation Dialogues """
def simulation_epoch(simulation_epoch_size):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in range(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        per_episode_reward = 0
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            per_episode_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("Simulation Episode %s: Success" % (episode))
                else:
                    print("Simulation Episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

        print("\tsimulation_epoch:", per_episode_reward)
        # cumulative_reward += user_sim.reward

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['avg_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['avg_turns'] = float(cumulative_turns) / simulation_epoch_size
    print("Func - \"simulation_epoch\":\n\tSimulation Success Rate %s\n\tAvg Reward %s\n\tAvg Turns %s" %
          (res['success_rate'], res['avg_reward'], res['avg_turns']), '\n')

    return res


""" Run Episodes """
def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    # if agt == 9 and params['trained_model_path'] == None and warm_start == 1:
    if warm_start == 1:
        print('Warm_start Starting ...\n')
        warm_start_simulation(warm_start_epochs)
        print('Warm_start Finished, Start RL Training ...\n')

    for episode in range(count):
        print("Episode: %s" % (episode))
        dialog_manager.initialize_episode()
        episode_over = False
        per_episode_reward = 0
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            per_episode_reward += reward
            if episode_over:
                if reward > 0:
                    print("Successful Dialog!")
                    successes += 1
                else:
                    print("Failed Dialog!")

                cumulative_turns += dialog_manager.state_tracker.turn_count

        print("\trun_episodes:", per_episode_reward)
        # cumulative_reward += user_sim.reward

    # simulation
    # if agt == 9 and params['trained_model_path'] == None:
        agent.predict_mode = True
        print("Get Simulation Results......")
        simulation_res = simulation_epoch(simulation_epoch_size)

        performance_records['success_rate'][episode] = simulation_res['success_rate']
        performance_records['avg_turns'][episode] = simulation_res['avg_turns']
        performance_records['avg_reward'][episode] = simulation_res['avg_reward']

        if simulation_res['success_rate'] >= best_res['success_rate']:
            if simulation_res['success_rate'] >= success_rate_threshold:  # threshold = 0.30
                agent.experience_replay_pool = [] # clear the exp-pool by better dialogues
                # print("simulation_res['success_rate'] >= best_res['success_rate']")
                simulation_epoch(simulation_epoch_size)

        if simulation_res['success_rate'] > best_res['success_rate']:
            # best_model['model'] = copy.deepcopy(agent)
            best_res['success_rate'] = simulation_res['success_rate']
            best_res['avg_reward'] = simulation_res['avg_reward']
            best_res['avg_turns'] = simulation_res['avg_turns']
            best_res['epoch'] = episode

        agent.clone_dqn = copy.deepcopy(agent.dqn)
        agent.train(batch_size, 1)
        agent.predict_mode = False

        print("Simulation Success Rate %s, Avg Reward %s, Avg Turns %s, Best Success Rate %s" % (
            performance_records['success_rate'][episode], performance_records['avg_reward'][episode], performance_records['avg_turns'][episode], best_res['success_rate']))

        # save the model every 10 episodes
        if episode % save_check_point == 0 and params['trained_model_path'] == None:
            # save_model(params['write_model_dir'], agt, best_res['success_rate'],
            #            best_model['model'], best_res['epoch'], episode)
            save_keras_model(params['write_model_dir'], agt, best_res['success_rate'],
                             best_res['epoch'], episode)
            save_performance_records(
                params['write_model_dir'], agt, performance_records)

        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.3f Avg turns: %.3f\n" %
                (episode + 1, count, successes, episode + 1, float(cumulative_reward) / (episode + 1), float(cumulative_turns) / (episode + 1)))

    print("Final Success rate: %s / %s Avg reward: %.3f Avg turns: %.3f" %
                (successes, count, float(cumulative_reward) / count, float(cumulative_turns) / count))
    status['successes'] += successes
    status['count'] += count

    # if agt == 9 and params['trained_model_path'] == None:
    # save_model(params['write_model_dir'], agt, float(successes) / count, best_model['model'], best_res['epoch'], count)
    save_keras_model(params['write_model_dir'], agt, float(successes) / count, best_res['epoch'], count)
    save_performance_records(params['write_model_dir'], agt, performance_records)


run_episodes(1000, status)

res = None
with open('./dqn_agent/checkpoints/agt_9_performance_records.json', 'r') as f:
    res = json.loads(f.readline())

plot_sr(res)
plot_ar(res)
plot_at(res)
