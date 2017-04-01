from crawler.models import *
import numpy as np
from .apps import *


def query_course(goal=None, **kwargs):
    """Return list of Course objects
        Where goal in ['instructor', 'title', 'classroom', 'schedule']
        1. 自然語言處理在哪裡上課?
        request.query_course(goal='classroom', title='自然語言處理') -> '資105'
        2. 陳信希老師有開什麼課?
        request.query_course(goal='title', instructor='陳信希') -> ['專題研究', '自然語言處理']
        3. 日文一有哪些老師開課?
        request.query_course(goal='instructor', title__contains='日文一') -> ['張鈞竹', '彭誼芝', '施姵伃', '星野風紗子', ...'黃佳鈴', '黃意婷']
        4. 機器學習技法是在什麼時候上課?
        request.query_course(goal='schedule', title='機器學習技法') -> '二5,6'
    """
    ans = []
    if kwargs:
        query_term = {}
        for k, v in kwargs.items():
            query_term[k + '__contains'] = v

        courses = Course.objects.filter(**query_term).filter(semester='105-2')
        for course in courses:
            eval('ans.append(course.%s)' % goal)

        return np.unique(ans)

    return '我聽不懂您說的話耶QQ'


def understand(input):
    tokens = [tok for tok in jieba.cut(input)]
    intent, tokens, labels = get_intent_slot(lu_model, tokens, word2idx, idx2label, idx2intent)

    print (tokens, labels, intent)
    d = {'tokens': tokens, 'labels': labels, 'intent': intent, 'slot': {}}
    for label, token in zip(labels, tokens):
        if label != 'O':
            d['slot'][label[2:]] = token

    return d
