# coding: utf-8
import numpy as np
from access_django import *
from utils.lu import *
from utils.nlg import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *
from utils.query import *

all_courses = list(query_course({}).values())
np.random.shuffle(all_courses)

f = open('mtlu_eval.log', 'w')
tot_reward = 0
correct = 0
tot_turn = 0
N = 100
MAX_BI_TURN = 5
DST_turn_acc = 0
DST_episode_acc = 0
user_sim = RuleSimulator(all_courses)

for i in range(N):
    uid = i
    user_action = user_sim.initialize_episode()

    prev_state = {'request_slots':{}, 'inform_slots':{}}
    for j in range(MAX_BI_TURN):
        tot_turn += 2
        print('User_act', user_action, file=f)
        user_sentence = user_action['nl']
        resp = {}
        resp['sementic'], resp['status'], resp['action'], resp['resp_str'] = multi_turn_lu3(uid, user_sentence)

        user_act_for_DST = {}
        user_act_for_DST['intent'] = user_action['diaact'] # compatible with LU & DST
        user_act_for_DST['slot'] = user_action['inform_slots'] # compatible with LU & DST
        #print(user_act_for_DST)
        true_state = DST_update(prev_state, user_act_for_DST)
        #print("DST state:", resp['status'])
        #print("true state:", true_state)
        if resp['status'] == true_state:
            DST_turn_acc += 1

        system_sentence = agent2nl(resp['action'])
        user_action, over = user_sim.next(resp['action'])
        print('User  : %s' % user_sentence, file=f)
        print('LU result', resp['sementic'], file=f)
        print('Sys_in', resp['status'], file=f)
        print('Sys_act', resp['action'], file=f)
        print('System: %s' % system_sentence, file=f)
        if over or j == MAX_BI_TURN-1:
            reward = user_sim.reward_function()
            tot_reward += reward
            correct += 1 if reward > 0 else 0
            print('Reward: %d\n==============' % reward, file=f)        
            
            if resp['status'] == true_state:
                DST_episode_acc += 1

            break

        prev_state = true_state.copy()


print('Average reward: %f' % (tot_reward / N), file=f)
print('Accuracy: %f (%d/%d)' % (correct / N, correct, N), file=f)
print('Average Turn: %f' % (tot_turn / N), file=f)

DST_turn_acc /= tot_turn
DST_episode_acc /= N
print('Tracking accuracy [turn]: %f' % (DST_turn_acc), file=f)
print('Tracking accuracy [episode]: %f' % (DST_episode_acc), file=f)
f.close()
