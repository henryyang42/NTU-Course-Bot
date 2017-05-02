import numpy as np
#import generate_template
import pandas as pd
import jieba
import random


def status2NLP(status, num_request, where, when, instructor, title) :

    lst_total = [where, when, instructor, title]
    # pihua_start = ['', '請問', '告訴我', '我想知道', '幫我找', '幫我查', '幫我查一下', '查一下']
    # pihua_end = ['', '謝謝', '感謝']

    question = ['', '呢', '嗎', '你知道嗎']

    course_query = ['有哪些課', '開哪些課', '有開哪些課', '教哪些課']
    instructor_query = ['有哪些', '是哪個', '是哪位']
    time_query = ['什麼時候', '在幾點', '在星期幾', '在禮拜幾', '幾點幾分']
    teacher = ['', '老師', '教授']

    # for test
    # status is a 4x5 list
    # status where when who what
    # status = [ confirm_or_not, misunderstood_or_not, inform_or_not, request_or_not, constraint_or_not]
    status = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0]
    t1 = '我想選星期二的課'
    t1_seg = ','.join(jieba.cut(t1)) # '我,想,選,星期二,的,課'
    status_after_t1 = [[0,0,0,0,0],[0,0,1,0,1],[0,0,0,0,1],[0,0,0,1,0]
    t2 = '有陳縕儂老師的課嗎'
    t2_seg = ','.join(jieba.cut(t2)) # '有,陳縕儂,老師,的,課,嗎'
    status_after_t1 = [[0,0,0,0,0],[1,0,1,0,1],[0,0,1,0,1],[0,0,1,1,0]

    flag = 0
    lst_NLP = []
    lst_BIO = []
    for i1 in range(num_request) :
        if status[i1][4] == 1 and status[i1][2] == 0:
            status[i1][2] = 1
            lst_NLP = lst_NLP.append(random.choice(pihua_start))
            lst_NLP = lst_NLP.append(random.choice(lst_total[i1]))

if __name__ == '__main__' :

    where = ['在哪', '哪裡', '在哪裡', '哪間教室', '在哪間教室', '在什麼地方', '哪個館', '哪個系館', '在哪個系館', '在哪個系館哪間教室']

    when = ['', '今天', '星期一', '星期二', '星期三', '星期四', '星期五', '禮拜一', '禮拜二', '禮拜三', '禮拜四', '禮拜五']

    titles = []
    instructors = []

    db = pd.read_csv('./db.csv').fillna('')
    titles = db.title.unique().tolist()
    instructors = db.title.unique().tolist()

    jieba.load_userdict('./entity_dictionary_2.txt')







'''
sim_status = np.zeros((4,4))

class sim_status :
    def __init__(self) :
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
    def randomlize(self) :
'''
