# coding: utf-8
import django
import os
import random
import numpy as np
import pandas as pd
import re

import sys
sys.path.append("../")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()

from crawler.const import base_url
from django.template import Context, Template
from crawler.models import *

import jieba
if not os.path.exists('dict.big.txt'):
    os.system("wget %s -O %s" % (
        'https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big',
        'dict.big.txt'))
jieba.set_dictionary('dict.big.txt')
random.seed(123)


def BIO(sentence, context):
    inv_context = {v: k for k, v in context.items()}
    toks = sentence.replace(' ', '')
    toks = list(jieba.cut(toks))
    tags = ['O'] * len(toks)
    for i, tok in enumerate(toks):
        tag = inv_context.get(tok, '')
        if tag in ['title', 'when', 'instructor', 'classroom']:
            tags[i] = 'B_%s' % tag

    return tags


# status = [[ confirm_or_not, misunderstood_or_not, inform_or_not, request_or_not, constraint_or_not] * 4 ]
# 4 for what who when where


def generate_sentence_auto_mode(status, course):
    sentence, sentence_seg, bio = '', '', ''
    # 4 * 2        2 for request, constraint      4 for what who when where
    status_for_MTLU = np.zeros((4, 2), dtype=int)
    for i, st in enumerate(status):
        if st[4] and not st[2]:
            # print('constraint_or_not', request[i])
            st[2] = 1
            status[i] = st
            status_for_MTLU[i][1] = 1
            status_for_MTLU = ' '.join([str(x)
                                        for x in status_for_MTLU.flatten()])
            #status_for_MTLU = ' '.join(status_for_MTLU.flatten().tolist())

            context = {request[i]: course[request[i]], 'where': random.choice(where), 'time_query': random.choice(
                time_query), 'course_query': random.choice(course_query), 'instructor_query': random.choice(instructor_query)}
            # print (context)
            tpl = random.choice(constraint_tpl[i])
            sentence = tpl.render(Context(context))
            #sentence = sentence_replace(sentence)
            # print (sentence)
            bio = ' '.join(BIO(sentence, context))
            # print (bio)
            sentence_seg = ' '.join(jieba.cut(sentence))
            return sentence_seg, bio, status, status_for_MTLU
    for i, st in enumerate(status):
        if st[3] and not st[2]:
            # print('request_or_not', request[i])
            st[2] = 1
            status[i] = st
            status_for_MTLU[i][0] = 1
            status_for_MTLU = ' '.join([str(x)
                                        for x in status_for_MTLU.flatten()])
            tpl = random.choice(request_tpl[i])
            sentence = tpl.render(Context({}))
            #sentence = sentence_replace(sentence)
            bio = ' '.join(BIO(sentence, {}))
            sentence_seg = ' '.join(jieba.cut(sentence))
            return sentence_seg, bio, status, status_for_MTLU
    # print('done')
    status_for_MTLU = ' '.join([str(x) for x in status_for_MTLU.flatten()])
    return sentence_seg, bio, status, status_for_MTLU


def status_maker():
    status = np.zeros((4, 5), dtype=int)
    if user_mode == 0:
        # print('auto_mode')
        flag1 = 1
        while flag1:
            for i, st in enumerate(status):
                st[3] = random.randint(0, 1)
                if i != 3 and not st[3]:
                    st[4] = random.randint(0, 1)
                    if st[4]:
                        flag1 = 0
                status[i] = st
    # print(status)
    return status



# pihua_start = ['', '請問', '告訴我', '我想知道', '幫我找', '幫我查', '幫我查一下', '查一下']
# pihua_end = ['', '謝謝', '感謝']
where = ['在哪', '哪裡', '在哪裡', '哪間教室', '在哪間教室',
         '在什麼地方', '哪個館', '哪個系館', '在哪個系館', '在哪個系館哪間教室']
# question = ['', '呢', '嗎', '你知道嗎']
when = ['今天', '星期一', '星期二', '星期三', '星期四',
        '星期五', '禮拜一', '禮拜二', '禮拜三', '禮拜四', '禮拜五']
course_query = ['有哪些課', '開哪些課', '有開哪些課', '教哪些課']
instructor_query = ['有哪些', '是哪個', '是哪位']
time_query = ['什麼時候', '在幾點', '在星期幾', '在禮拜幾', '幾點幾分']
# teacher = ['', '老師', '教授']
titles = []
instructors = []
courses = []
classrooms = []
all_course = list(Course.objects.filter(semester='105-2'))


def trim_attr(s):
    s = re.sub(r'\（[^)]*\）', '', s)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\-[^)]*', '', s)
    for _ in range(3):
        if s and s[-1] in '一二三四五六上下':
            s = s[:-1]
    for rep in ' ()（）：:-「」《》、/+':
        s = s.replace(rep, '')

    return s

for course in all_course:
    titles.append(trim_attr(course.title))
    classrooms.append(trim_attr(course.classroom))
    instructors.append(course.instructor)
    if course.instructor and course.schedule_str:
        courses.append({'title': course.title, 'instructor': course.instructor, 'when': '星期' + course.schedule_str[:1], 'classroom': course.classroom})


titles = np.unique([x for x in titles if x and ' ' not in x])
instructors = np.unique([x for x in instructors if x and ' ' not in x])
classrooms = np.unique([x for x in classrooms if x and ' ' not in x])
lst_dict = []

lst_dict.extend(titles)
lst_dict.extend(instructors)
lst_dict.extend(when)
lst_dict.extend(classrooms)
lst_dict.extend(['星期幾', '禮拜幾'])
with open('entity_dictionary_2_replace.txt', 'w') as f:
    f.write('\n'.join(['%s 99999' % s for s in lst_dict]))
f.close()
jieba.load_userdict('./entity_dictionary_2_replace.txt')
print ('%d titles, %d instructors' % (len(titles), len(instructors)))

template_folder = 'MTLU_template'
os.makedirs(template_folder, exist_ok=True)


user_mode = 0
# what who when where
# title instructor when classroom
constraint_tpl = [
    [Template('{{title}}'), Template('課名是{{title}}'), Template('課程名稱為{{title}}')],
    [Template('{{instructor}}'), Template('老師是{{instructor}}'), Template('有沒有{{instructor}}老師的課'), Template('教學老師是{{instructor}}'), Template('幫我查{{instructor}}老師的課'), Template('{{instructor}}上的課'), Template('是{{instructor}}教授'), Template('{{instructor}}老師')],
    [Template('{{when}}'), Template('上課時間是{{when}}'), Template('我想上{{when}}的課'), Template('我想選{{when}}的課'), Template('有沒有{{when}}的課')],
    [Template('{{classroom}}'), Template('上課教室是{{classroom}}'), Template('在{{classroom}}上課'), Template('在{{classroom}}')]
]
request_tpl = [
    [Template('請列出課程名稱'), Template('什麼課')],
    [Template('老師的名字'), Template('老師是誰')],
    [Template('這堂課在星期幾上課?'), Template('上課時間在什麼時候'), Template('上課時間')],
    [Template('在哪裡上課'), Template('教室在哪')]]

# status what who when where
# status = [ confirm_or_not, misunderstood_or_not, inform_or_not, request_or_not, constraint_or_not]



#status = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0]]
request = ['title', 'instructor', 'when', 'classroom']
#content = [titles, instructors, when, []]


len_history = 4
num_sample = 250
# shape = (?*4, 6)
# ['index_sample', 'index_turn', 'sentence', 'BIO', 'status', 'status_for_MTLU']
df_log = pd.DataFrame([], columns=['index_sample', 'index_turn',
                                   'sentence', 'BIO', 'status', 'status_for_MTLU'])
for j in range(num_sample):
    print(j)
    status = status_maker()
    course = random.choice(courses)
    for i in range(len_history):
        s, bio, status, status_for_MTLU = generate_sentence_auto_mode(status, course)
        str_status = ' '.join([str(x) for x in status.flatten()])
        df_temp = pd.DataFrame({'index_sample': [j],
                                'index_turn': [i],
                                'sentence': [s],
                                'BIO': [bio],
                                'status': [str_status],
                                'status_for_MTLU': [status_for_MTLU]})
        df_log = df_log.append(df_temp, ignore_index=True)

df_log.to_csv('./MTLU_template/simmulator_log.csv')
