# coding: utf-8
try:
    from .access_django import *
except:
    from access_django import *
import random
from django.template import Context, Template
from utils.nlg import *
from utils.tagger import *
from utils.query import *
import sys

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
be = ['', '是']
ask = ['', '請問', '請告訴我', '請跟我說', '我需要知道']
q_end = ['', '?', '？']

templates = {
    'request_title': [
        Template('{{ask}}課程名稱是{{q_end}}'),
        Template('{{ask}}課名{{q_end}}'),
        Template('{{ask}}是哪一門課{{q_end}}'),
        Template('{{ask}}要找哪一門課{{q_end}}'),
    ],
    'request_instructor': [
        Template('{{ask}}老師是誰{{q_end}}'),
        Template('{{ask}}老師是哪位{{q_end}}'),
        Template('{{ask}}是誰開的{{q_end}}'),
        Template('{{ask}}哪位教授開的{{q_end}}'),
        Template('{{ask}}教授是誰{{q_end}}'),
        Template('{{ask}}是誰教的{{q_end}}'),
    ],
    'request_schedule_str': [
        Template('{{ask}}什麼時候的課{{q_end}}'),
        Template('{{ask}}上課時間在什麼時候{{q_end}}'),
        Template('{{ask}}什麼時候上課{{q_end}}'),
        Template('{{ask}}星期幾上課{{q_end}}'),
        Template('{{ask}}禮拜幾上課{{q_end}}'),
    ],
    'request_designated_for': [
        Template('{{ask}}是哪個系開的{{q_end}}'),
        Template('{{ask}}是哪個系所開的{{q_end}}'),
        Template('{{ask}}什麼系開的{{q_end}}'),
        Template('{{ask}}什麼系所的課{{q_end}}'),
        Template('是哪個系的課{{q_end}}'),
        Template('是哪個系所的課{{q_end}}'),
    ],
    'inform_base': [
        Template('流水號{{serial_no}}:{{title}}，授課教師是{{instructor}}。'),
        Template('流水號{{serial_no}}:{{instructor}}開的{{title}}。'),
        Template('{{serial_no}}:{{title}} by {{instructor}}。'),
        Template('[{{serial_no}}]{{instructor}}開授的{{title}}。'),
    ],
    'inform_when': [
        Template('{{be}}{{when}}。'),
        Template('{{be}}{{when}}的課。'),
        Template('{{be}}{{when}}上課。'),
        Template('在{{when}}上課。'),
        Template('{{be}}{{when}}上的。'),
    ],
    'inform_classroom': [
        Template('{{be}}{{classroom}}。'),
        Template('在{{classroom}}上課。'),
        Template('上課地點是{{classroom}}。'),
        Template('教室在{{classroom}}。'),
    ],
    'inform_designated_for': [
        Template('{{be}}{{designated_for}}開的。'),
        Template('{{be}}{{designated_for}}的課。'),
        Template('{{be}}{{designated_for}}上的。'),
    ],
    'inform_sel_method':[ 
        Template('{{be}}{{sel_method}}。'),
        Template('加選方法{{be}}{{sel_method}}。'),
        Template('加簽方式{{be}}{{sel_method}}。'),
        Template('選課方法{{be}}{{sel_method}}。'),
        Template('{{be}}{{sel_method}}。'),
    ],
    'thanks': [
        Template('謝謝'),
        Template('感謝你'),
        Template('感恩'),
    ],
    'closing': [
        Template('不好意思，沒有找到符合條件的課程。'),
        Template('不好意思，可以重新說一次您的條件嗎?'),
        Template('沒有幫您找到符合條件的課程，很抱歉。'),
        Template('我找不到這樣的課耶QQ'),
    ],
}


def trim_course(course):
    course['when'] = random.choice(['星期', '禮拜']) + course['schedule_str'][0]
    course['be'] = random.choice(be)
    course['ask'] = random.choice(ask)
    for k in ['title', 'instructor', 'classroom']:
        course[k] = trim_attr(course[k])
    return course

if __name__ == '__main__':
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
    print('[Info] Start generating templates')
    # TODO Change to argparse
    #filename = 'training_template.txt'
    filename = sys.argv[1]
    N = 100
    courses = query_course({}).values()  # Get all course
    # TODO Refine request_schedule_str to when
    #
    with open(filename, 'w') as f:
        for intent, tpls in templates.items():
            for tpl in tpls:
                for _ in range(N):
                    course = random.choice(courses)
                    course = trim_course(course)
                    # Jieba cut sentence
                    sentence = ' '.join(cut(tpl.render(Context(course))))
                    # BIO tagged sentence
                    bio_tagged = ' '.join(BIO(sentence, course))

                    f.write(intent + '\n')
                    f.write(sentence + '\n')
                    f.write(bio_tagged + '\n')
