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

templates = {
    'request_title': [
        Template('請告訴我課名。'),
        Template('你知道課程名稱嗎?'),
    ],
    'request_instructor': [
        Template('是哪一位老師開的課呢?'),
        Template('請問您要找的是誰開的?'),
    ],
    'request_schedule_str': [
        Template('您希望的上課時間在哪個時段?'),
        Template('您想找星期幾上課的?'),
    ],
    'request_designated_for': [
        Template('可以告訴我開課的是哪個系嗎?'),
        Template('請問您是要找哪個系的呢?'),
    ],
    'inform_base': [
        Template('流水號{{serial_no}}:{{title}}，由{{instructor}}老師授課。'),
    ],
    'inform_when': [
        Template('上課的時間是{{when}}喔!'),
        Template('是在{{when}}上課。'),
    ],
    'inform_classroom': [
        Template('是在{{classroom}}上課。'),
        Template('上課教室是{{classroom}}。'),
    ],
    'inform_designated_for': [
        Template('開課的系所是{{designated_for}}。'),
        Template('這門課是{{designated_for}}開的課。'),
    ],
    'inform_required_elective': [
        Template('是是一門{{required_elective}}。'),
        Template('這門課是{{required_elective}}課。'),
    ],
    'inform_sel_method':[ 
        Template('要加簽的話方法是{{sel_method}}。'),
        Template('這門課是{{sel_method}}類加選。'),
    ],
    'thanks': [
        Template('謝謝，請問還需要什麼服務嗎?'),
        Template('感謝您的使用。'),
    ],
    'closing': [
        Template('不好意思，查詢不到符合您條件的課程。'),
        Template('我沒有找到符合條件的課，可以重新說一次您的條件嗎?'),
    ],
}


def trim_course(course):
    course['when'] = random.choice(['星期', '禮拜']) + course['schedule_str'][0]
    #course['be'] = random.choice(be)
    #course['ask'] = random.choice(ask)
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
    N = 10
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
