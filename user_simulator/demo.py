import argparse, json, copy, os, random
try:
    import cPickle as pickle
except ImportError:
    import pickle

from .manager import DialogManager
from .usersim.usersim_rule import RuleSimulator

import django
from crawler.const import base_url
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()

from django.db.models.aggregates import Count
from django.db.models import Q
from django.template import Context, Template
from crawler.models import *




def usim_initial():

    all_courses = Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2').all().values()[:100]

    user_sim = RuleSimulator(all_courses)
    dialog_manager = DialogManager(None, user_sim)
    user_action = dialog_manager.initialize_episode()

    pickle.dump(dialog_manager, open('user_simulator/log/dm.p', 'wb'))

    possible_answer = dialog_manager.possible_answer[dialog_manager.query_slot]
    possible_num = dialog_manager.possible_answer['count']

    return user_sim.goal, user_action, possible_num, possible_answer

def usim_request(request):

    # Log  QQ
    dialog_manager = pickle.load(open('user_simulator/log/dm.p', 'rb'))

    request['request_slots'] = {k:v for k, v in request['request_slots'].items() if v}

    dialog_manager.sys_action = request
    episode_over, reward = dialog_manager.next_turn()
    agent_action = dialog_manager.sys_action
    user_action = dialog_manager.user_action

    pickle.dump(dialog_manager, open('user_simulator/log/dm.p', 'wb'))

    possible_answer = dialog_manager.possible_answer[dialog_manager.query_slot]
    possible_num = dialog_manager.possible_answer['count']

    response = {'agent':agent_action, 'user':user_action, 'num':possible_num, 'suggest':possible_answer}
    
    print(reward)

    return response


