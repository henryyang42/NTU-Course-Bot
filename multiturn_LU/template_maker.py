# coding: utf-8
import django
import os
import random
import numpy as np
import pandas as pd


import sys
#sys.path.append("/Users/xogo/Desktop/NTU/2017_spring/ICB/")
sys.path.append("../")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()

from crawler.const import base_url
from django.template import Context, Template
from crawler.models import *

import jieba
jieba.load_userdict('./entity_dictionary_2.txt')

def formulate(**kwargs):
    kwlist = list(kwargs.items())
    len_kw = len(kwlist)
    state = [0] * (len_kw + 1)

    total = 1
    for _, l in kwlist:
        total *= len(l)

    print ('Formulating %d sentences...' % total)

    while True:
        d = {}
        for i, v in enumerate(state[:-1]):
            d[kwlist[i][0]] = kwlist[i][1][v]
        yield d

        state[0] += 1
        ind = 0

        while ind < len_kw and state[ind] == len(kwlist[ind][1]):
            state[ind] = 0
            state[ind + 1] += 1
            ind += 1

        if len_kw == ind:
            break

def BIO(sentence, context):
    inv_context = {v: k for k, v in context.items()}
    #toks = sentence.split()
    toks = sentence.replace(' ','')
    toks = list(jieba.cut(toks))
    tags = ['O'] * len(toks)
    for i, tok in enumerate(toks):
        tag = inv_context.get(tok, '')
        if tag in ['title', 'when', 'instructor']:
            tags[i] = 'B_%s' % tag

    #return ' '.join(tags) + '\n'
    return tags

#def status_for_MTLU() :

# status = [[ confirm_or_not, misunderstood_or_not, inform_or_not, request_or_not, constraint_or_not] * 4 ]
# 4 for what who when where
def generate_sentence_auto_mode(status):
    sentence, sentence_seg, bio = '', '', ''
    status_for_MTLU = np.zeros((4,2),dtype=int) # 4 * 2        2 for request, constraint      4 for what who when where
    for i, st in enumerate(status):
        if st[4] and not st[2]:
            print('constraint_or_not', request[i])
            st[2] = 1
            status[i] = st
            status_for_MTLU[i][1] = 1
            status_for_MTLU = ' '.join([str(x) for x in status_for_MTLU.flatten()])
            #status_for_MTLU = ' '.join(status_for_MTLU.flatten().tolist())
            context = {request[i]: content[i][0]}
            print (context)
            sentence = constraint_tpl[i].render(Context(context))
            print (sentence)
            bio = ' '.join(BIO(sentence, context))
            print (bio)
            sentence_seg = ' '.join(jieba.cut(sentence))
            return sentence_seg, bio, status, status_for_MTLU
    for i, st in enumerate(status):
        if st[3] and not st[2]:
            print('request_or_not', request[i])
            st[2] = 1
            status[i] = st
            status_for_MTLU[i][0] = 1
            status_for_MTLU = ' '.join([str(x) for x in status_for_MTLU.flatten()])
            sentence = request_tpl[i].render(Context({}))
            bio = ' '.join(BIO(sentence, {}))
            sentence_seg = ' '.join(jieba.cut(sentence))
            return sentence_seg, bio, status, status_for_MTLU
    print('done')
    status_for_MTLU = ' '.join([str(x) for x in status_for_MTLU.flatten()])
    return sentence_seg, bio, status, status_for_MTLU

def generate_sentence_user_mode(status):
    print('not finished')
    sentence, bio = '', ''
    for i, st in enumerate(status):
        if st[4] and not st[2]:
            print('constraint_or_not', request[i])
            st[2] = 1
            context = {request[i]: content[i][0]}
            print (context)
            sentence = constraint_tpl[i].render(Context(context))
            print (sentence)
            bio = BIO(sentence, context)
            print (bio)
            return sentence, bio, status
    for i, st in enumerate(status):
        if st[3] and not st[2]:
            print('request_or_not', request[i])
            st[2] = 1
            sentence = request_tpl[i].render(Context({}))
            bio = BIO(sentence, {})
            return sentence, bio, status
    print('done')
    return sentence, bio, status





# pihua_start = ['', '請問', '告訴我', '我想知道', '幫我找', '幫我查', '幫我查一下', '查一下']
# pihua_end = ['', '謝謝', '感謝']
where = ['在哪', '哪裡', '在哪裡', '哪間教室', '在哪間教室', '在什麼地方', '哪個館', '哪個系館', '在哪個系館', '在哪個系館哪間教室']
# question = ['', '呢', '嗎', '你知道嗎']
when = ['今天', '星期一', '星期二', '星期三', '星期四', '星期五', '禮拜一', '禮拜二', '禮拜三', '禮拜四', '禮拜五']
course_query = ['有哪些課', '開哪些課', '有開哪些課', '教哪些課']
instructor_query = ['有哪些', '是哪個', '是哪位']
time_query = ['什麼時候', '在幾點', '在星期幾', '在禮拜幾', '幾點幾分']
# teacher = ['', '老師', '教授']
titles = []
instructors = []

all_course = list(Course.objects.filter(semester='105-2'))

for course in all_course:
    titles.append(course.title)
    instructors.append(course.instructor)

titles = np.unique([x for x in titles if x and ' ' not in x])
instructors = np.unique([x for x in instructors if x and ' ' not in x])

print ('%d titles, %d instructors' % (len(titles), len(instructors)))

template_folder = 'MTLU_template'
os.makedirs(template_folder, exist_ok=True)



user_mode = 0
#constraint_tpl = [Template('我想找 {{classroom}} 的 課\n'), Template('我想找 {{when}} 的 課\n'), Template('我想找 {{instructor}} 的 課\n'), Template('我想找 {{title}} 的 課\n')]
#request_tpl = [Template('在哪裡上課\n'), Template('這堂課在星期幾 ?\n'), Template('有什麼課\n'), Template('老師是誰\n')]

# what who when where
# title instructor when classroom
constraint_tpl = [Template('課名是{{title}}'), Template('教學老師是{{instructor}}'), Template('上課時間是{{when}}'), Template('上課教室是{{classroom}}')]
request_tpl = [Template('請列出課程名稱'), Template('老師的名字'), Template('這堂課在星期幾上課?'), Template('在哪裡上課') ]
# status what who when where
# status = [ confirm_or_not, misunderstood_or_not, inform_or_not, request_or_not, constraint_or_not]
status = np.zeros((4,5),dtype=int)
print(status)
if user_mode == 0 :
    print('auto_mode')
    flag1 = 1
    while flag1 :
        for i, st in enumerate(status) :
            st[3] = random.randint(0,1)
            if i != 3 and not st[3]:
                st[4] = random.randint(0,1)
                if st[4] :
                    flag1 = 0
            status[i] = st
print(status)
'''
if user_mode == 0 :
    print('auto_mode')
    flag1 = 1
    flag2 = 1
    while flag1 and flag2 :
        for i, st in enumerate(status) :
            st[3] = random.randint(0,1)
            if i != 0 and not st[3]:
                st[4] = random.randint(0,1)
                if st[4] :
                    flag1 = 0
            if st[3] or st[4] :
                st[2] = random.randint(0,1)
            status[i] = st
        for i, st in enumerate(status) :
            if not st[2] and (st[3] or st[4]) :
                flag2 = 0
'''



#status = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0]]
request = ['title', 'instructor', 'when', 'classroom']
content = [ titles, instructors, when, []]


len_history = 4
num_sample = 1
# shape = (?*4, 6)
# ['index_sample', 'index_turn', 'sentence', 'BIO', 'status', 'status_for_MTLU']
df_log = pd.DataFrame([], columns=['index_sample', 'index_turn', 'sentence', 'BIO', 'status', 'status_for_MTLU'])
for j in range(num_sample) :
    for i in range(len_history) :
        s, bio, status, status_for_MTLU = generate_sentence_auto_mode(status)
        str_status = ' '.join([str(x) for x in status.flatten()])
        df_temp = pd.DataFrame({'index_sample' : [j],
                            'index_turn' : [i],
                            'sentence' : [s],
                            'BIO' : [bio],
                            'status' : [str_status],
                            'status_for_MTLU' : [status_for_MTLU]})
        df_log = df_log.append(df_temp,ignore_index=True)

df_log.to_csv('./MTLU_template/simmulator_log.csv')
'''
print (df_log)
print(status)
s, bio, status, status_for_MTLU = generate_sentence_auto_mode(status)
print(s, status, bio, status_for_MTLU)
s, bio, status, status_for_MTLU = generate_sentence_auto_mode(status)
print(s, status, bio, status_for_MTLU)
s, bio, status, status_for_MTLU = generate_sentence_auto_mode(status)
print(s, status, bio, status_for_MTLU)
s, bio, status, status_for_MTLU = generate_sentence_auto_mode(status)
print(s, status, bio, status_for_MTLU)
'''
