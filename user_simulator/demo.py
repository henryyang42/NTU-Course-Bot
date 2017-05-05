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

from utils.nlg import sem2nl, agent2nl

def usim_initial():

    # Initialize Course set
    all_courses = Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2').all().values()[:100]

    # Initialize simulator
    user_sim = RuleSimulator(all_courses)
    dialog_manager = DialogManager(None, user_sim)
    user_action = dialog_manager.initialize_episode()

    # Dump system status
    pickle.dump(dialog_manager, open('user_simulator/dm.p', 'wb'))

    # Suggest Possible Answers
    possible_answer = dialog_manager.possible_answer[dialog_manager.query_slot]
    possible_num = dialog_manager.possible_answer['count']

    # Add Natural language
    user_action['nl'] = sem2nl(user_action)

    return user_sim.goal, user_action, possible_num, possible_answer

def usim_request(request):

    # Load system status
    dialog_manager = pickle.load(open('user_simulator/dm.p', 'rb'))

    # Remove blank slots
    request['request_slots'] = {k:v for k, v in request['request_slots'].items() if v}

    if request['diaact'] == 'closing':
        agent_action = request
        user_action = dialog_manager.initialize_episode()
        episode_over = True

    else:
        #
        dialog_manager.sys_action = request
        episode_over = dialog_manager.next_turn()
        agent_action = dialog_manager.sys_action
        user_action = dialog_manager.user_action

    # Suggest Possible Answers
    possible_answer = dialog_manager.possible_answer[dialog_manager.query_slot]
    possible_num = dialog_manager.possible_answer['count']

    turn = user_action['turn']

    response = [
        [ "SYS Turn "+ str(turn-1), agent_action['diaact'], agent2nl(agent_action)],
        [ "Possible values:", possible_num, possible_answer],
        [ "USR Turn "+str(turn), user_action['diaact'], sem2nl(user_action)],
    ]

    if episode_over :
        if user_action['diaact'] == 'deny':
            response.append(["Reward:", -100-turn])
            dialog_manager.reward = dialog_manager.reward - 100 - turn
        elif user_action['diaact'] == 'thanks':
            response.append(["Reward:", 100-turn])
            dialog_manager.reward = dialog_manager.reward + 100 - turn
        else:
            pass

        response.append(
            ["Total episodes: " + str(dialog_manager.episode_times-1), 
             "Correct times: "+str(dialog_manager.episode_correct),
             "Accumulate reward: " + str(dialog_manager.reward),
            ]
        )
        user_action = dialog_manager.initialize_episode()
        response.append([ "New Turn!"])
        response.append([ "USR Turn "+str(user_action['turn']), user_action['diaact'], sem2nl(user_action)])

    # Dump system status
    pickle.dump(dialog_manager, open('user_simulator/dm.p', 'wb'))

    return json.dumps(response)


