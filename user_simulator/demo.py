import argparse, json, copy, os, random
try:
    import cPickle as pickle
except ImportError:
    import pickle

from .usersim.usersim_rule import RuleSimulator
from utils.query import query_course

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

    # Initialize user simulator
    user = RuleSimulator(all_courses)
    user_action = user.initialize_episode()

    # Dump system status
    pickle.dump(user, open('user_simulator/dm.p', 'wb'))

    # Suggest Possible Answers
    request_slot = user.request_slot
    answer_set = query_course(user.state['history_slots']).values_list(request_slot, flat=True)
    possible_answer = {request_slot:answer_set,'count':len(answer_set)}

    # Add Natural language
    user_action['nl'] = sem2nl(user_action)

    return user.goal, user_action, possible_answer['count'], possible_answer[request_slot]

def usim_request(request):

    # Load system status
    user = pickle.load(open('user_simulator/dm.p', 'rb'))

    # Remove blank slots
    request['request_slots'] = {k:v for k, v in request['request_slots'].items() if v}

    #
    user_action, episode_over = user.next(request)
    agent_action = request

    # Suggest Possible Answers
    request_slot = user.request_slot
    answer_set = query_course(user.state['history_slots']).values_list(request_slot, flat=True)
    possible_answer = {request_slot:answer_set,'count':len(answer_set)}

    response = [
        [ "SYS Turn "+ str(user.state['turn']-1), agent_action['diaact'], agent2nl(agent_action)],
        [ "Possible values:", possible_answer['count'], possible_answer[request_slot][0:10]],
        [ "USR Turn "+str(user.state['turn']), user_action['diaact'], sem2nl(user_action)],
    ]

    # Calculate Reward
    if episode_over :

        reward, acc_reward = user.episodes_reward()
        episode_times, correct_times = user.episodes_times()

        response.append(["Reward:", reward])
        response.append(
            [
             "Total episodes: " + str(episode_times), 
             "Correct times: " + str(correct_times),
             "Accumulate reward: " + str(acc_reward+reward),
            ]
        )

        # New episode
        user_action = user.initialize_episode()
        # Suggest Possible Answers
        request_slot = user.request_slot
        answer_set = query_course(user.state['history_slots']).values_list(request_slot, flat=True)
        possible_answer = {request_slot:answer_set,'count':len(answer_set)}

        response.append([ "New Turn!"])
        response.append([ "----------","----------","----------------------------------------"])
        response.append([ "Possible values:", possible_answer['count'], possible_answer[request_slot][0:10]])
        response.append([ "USR Turn "+str(user_action['turn']), user_action['diaact'], sem2nl(user_action)])


    # Dump system status
    pickle.dump(user, open('user_simulator/dm.p', 'wb'))

    return json.dumps(response)


