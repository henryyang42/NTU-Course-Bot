# coding: utf-8
from access_django import *
import random
import numpy as np
from django.template import Context, Template
from utils.nlg import *
from utils.tagger import *
from utils.query import *
# Templates
# possible_slots = [
# 'title',
# 'when',
# 'instructor',
# 'classroom',
# 'designated_for',
# 'required_elective',
# 'sel_method'
# ]

templates = {
    'request_title': [
        Template('有開哪些課'),
        Template('有哪些課'),
        Template('{{instructor}}在{{when}}有哪些課')
    ],
    'request_instructor': [
        Template('老師是誰'),
        Template('老師是哪位'),
        Template('開{{title}}的老師是哪位'),
    ],
    'request_schedule_str': [
        Template('什麼時候的課'),
        Template('上課時間在什麼時候')
    ],
    'request_classroom': [
        Template('在哪裡上課'),
        Template('教室在哪')
    ],
    'request_review': [
        Template(''),
    ],
    'request_designated_for': [
        Template(''),
    ],
    'request_required_elective': [
        Template(''),
    ],
    'request_sel_method': [
        Template(''),
    ],
    'inform': [
        Template(''),
    ],
    'other': [
        Template(''),
    ]
}

if __name__ == '__main__':
    print('[Info] Start generating templates')
    """
    In this format:
    Line1: Intent
    Line2: Tokenized sentence
    Line3: BIO
    ======
    classroom
    禮拜三 高分子材料概論 在 哪個 系館 哪間 教室 上課 嗎 ?
    B_when B_title O O O O O O O O
    title
    幫 我 找 吳俊傑 教 哪些 課 ?
    O O O B_instructor O O O O
    """
    # TODO Change to argparse
    filename = 'training_template.txt'
    N = 10
    courses = query_course({}).values()  # Get all course
    # TODO Refine request_schedule_str to when
    #
    with open(filename, 'w') as f:
        for intent, tpls in templates.items():
            tpl = random.choice(tpls)
            course = random.choice(courses)
            # Jieba cut sentence
            sentence = ' '.join(cut(tpl.render(Context(course))))
            # BIO tagged sentence
            bio_tagged = ' '.join(BIO(sentence, course))

            f.write(intent)
            f.write(sentence)
            f.write(bio_tagged)
