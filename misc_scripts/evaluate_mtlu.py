# coding: utf-8
import numpy as np
from access_django import *
from utils.lu import multi_turn_lu2
from utils.nlg import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *


all_courses = list(Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2').all().values())
np.random.shuffle(all_courses)

f = open('mtlu_eval.log', 'w')
tot_reward = 0
correct = 0
tot_turn = 0
N = 100
user_sim = RuleSimulator(all_courses)
for i in range(N):
    uid = i
    user_action = user_sim.initialize_episode()

    for j in range(4):
        tot_turn += 2
        #print(user_action, file=f)
        #user_sentence = sem2nl(user_action)
        user_sentence = user_action['nl']
        resp = {}
        resp['sementic'], resp['status'], resp['action'], resp['resp_str'] = multi_turn_lu2(uid, user_sentence)
        system_sentence = agent2nl(resp['action'])
        user_action, over = user_sim.next(resp['action'])
        print('User  : %s' % user_sentence, file=f)
        #print(resp['action'], file=f)
        print('System: %s' % system_sentence, file=f)
        if over or j == 3:
            reward = user_sim.reward_function()
            tot_reward += reward
            correct += 1 if reward > 0 else 0
            print('Reward: %d\n==============' % reward, file=f)
            break


print('Average reward: %f' % (tot_reward / N), file=f)
print('Accuracy: %f (%d/%d)' % (correct / N, correct, N), file=f)
print('Average Turn: %f' % (tot_turn / N), file=f)
f.close()
