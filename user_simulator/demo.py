import argparse, json, copy, os, random
try:
    import cPickle as pickle
except ImportError:
    import pickle

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN, AgentDemo
from deep_dialog.usersims import RuleSimulator
from deep_dialog import dialog_config
from deep_dialog.dialog_config import *
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

import django
from crawler.const import base_url
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()

from django.db.models.aggregates import Count
from django.db.models import Q
from django.template import Context, Template
from crawler.models import *


""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""


def suggest(ans_dict):

    if ans_dict:
        ans = sum(ans_dict.values(),[])
        return ans, len(ans)
    return [], 0




def usim_initial():

    all_course = Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2').all()
    count = all_course.count()
    course = all_course[random.randint(0,count-1)]

    act_set = text_to_dict('deep_dialog/data/dia_acts.txt')
    slot_set = text_to_dict('deep_dialog/data/slot_set.txt')

    goal = {'diaact':'request','inform_slots':{}, 'request_slots':{}}
    goal['inform_slots'] = {'title':course.title, 'instructor':course.instructor,'classroom':course.classroom,'schedule_str':course.schedule_str}
    goal['request_slots'] = {'serial_no':'UNK'}
    goal_set = {'all':[goal]}

    agent_params = {'max_turn':40, 'epsilon':0, 'agent_run_mode':3, 'agent_act_level':0, 'cmd_input_mode':0}
    usersim_params = {'max_turn':40, 'slot_err_probability':0, 'slot_err_mode':0, 'intent_err_probability':0, 'simulator_run_mode':0, 'simulator_act_level':0, 'learning_phase':'all'}


    agent = AgentDemo(act_set, slot_set, agent_params)
    user_sim = RuleSimulator(act_set, slot_set, goal_set, usersim_params)
    dialog_manager = DialogManager(agent, user_sim, act_set, slot_set)
    user_action = dialog_manager.initialize_episode()

    pickle.dump(dialog_manager, open('user_simulator/log/dm.p', 'wb'))

    possible_answer, possible_num = suggest(dialog_manager.possible_answer)

    return goal, user_action, possible_num, possible_answer

def usim_request(request):

    # Log  QQ
    dialog_manager = pickle.load(open('user_simulator/log/dm.p', 'rb'))

    request['request_slots'] = {k:v for k, v in request['request_slots'].items() if v}

    dialog_manager.agent.request_cmd = request
    episode_over, reward, agent_action, user_action = dialog_manager.next_turn()

    pickle.dump(dialog_manager, open('user_simulator/log/dm.p', 'wb'))

    possible_answer, possible_num = suggest(dialog_manager.possible_answer)

    response = {'agent':agent_action, 'user':user_action, 'num':possible_num, 'suggest':possible_answer}

    return response


