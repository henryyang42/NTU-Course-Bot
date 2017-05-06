# coding: utf-8
from misc_scripts import access_django
from utils.lu import multi_turn_lu2
from utils.nlg import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from crawler.models import *
import time

all_courses = Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2')[:2].all().values()


for i in range(1):
    uid = i
    user_sim = RuleSimulator(all_courses)
    user_action = user_sim.initialize_episode()

    for _ in range(3):
        user_sentence = agent2nl(user_action)
        resp = {}
        resp['sementic'], resp['status'], resp['action'], resp['resp_str'] = multi_turn_lu2(uid, user_sentence)
        system_sentence = sem2nl(resp['action'])
        user_action = user_sim.next(resp['action'])[0]
        print('User  : %s' % user_sentence)
        print('System: %s' % system_sentence)
        time.sleep(1)

