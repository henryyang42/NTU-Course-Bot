# coding: utf-8
from misc_scripts import access_django
from utils.lu import multi_turn_lu2
from utils.nlg import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *
import time

all_courses = Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2')[:2].all().values()

f = open('mtlu_eval.log', 'w')
tot_reward = 0
N = 100
for i in range(N):
    uid = i
    user_sim = RuleSimulator(all_courses)
    user_action = user_sim.initialize_episode()

    for j in range(4):
        user_sentence = sem2nl(user_action)
        resp = {}
        resp['sementic'], resp['status'], resp['action'], resp['resp_str'] = multi_turn_lu2(uid, user_sentence)
        system_sentence = agent2nl(resp['action'])
        user_action, over = user_sim.next(resp['action'])
        print('User  : %s' % user_sentence, file=f)
        print('System: %s' % system_sentence, file=f)
        if over or j == 3:
            tot_reward += user_sim.reward_function()
            print('Reward: %d\n==============' % user_sim.reward_function(), file=f)
            break

print('Average reward: %f' % (tot_reward / N), file=f)
