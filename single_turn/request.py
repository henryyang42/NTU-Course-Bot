from crawler.models import *
import numpy as np
from .apps import *
from django.db.models import Q


def expand_title(title):
    q = Q()
    for c in title:
        q &= Q(title__contains=c)
    return q


def query_course(goal, slot):
    """Return list of Course objects
        Where goal in ['instructor', 'title', 'classroom', 'schedule']
        1. 自然語言處理在哪裡上課?
        query_course(goal='classroom', title='自然語言處理') -> '資105'
        2. 陳信希老師有開什麼課?
        query_course(goal='title', instructor='陳信希') -> ['專題研究', '自然語言處理']
        3. 日文一有哪些老師開課?
        query_course(goal='instructor', title__contains='日文一') -> ['張鈞竹', '彭誼芝', '施姵伃', '星野風紗子', ...'黃佳鈴', '黃意婷']
        4. 機器學習技法是在什麼時候上課?
        query_course(goal='schedule', title='機器學習技法') -> '二5,6'
    """

    # Transform slot to query terms.
    query_term = {}
    for k, v in slot.items():
        if k == 'when':
            query_term['schedule_str__contains'] = v[-1]
        elif k == 'title':
            pass
        else:
            query_term[k + '__contains'] = v

    # Generate corresponding response to each intent.
    courses = Course.objects.filter(expand_title(slot.get('title', ''))).filter(**query_term).filter(semester='105-2')
    print (courses)
    if not slot or courses.count() == 0:
        return [], '並未找到相符的課程。'
    if len(courses) > 20:
        courses = courses[:20]

    resp_list, resp_str = [], ''
    if goal == 'instructor':
        courses = [c for c in courses if c.instructor != '']
        resp_list = list(np.unique([course.instructor for course in courses]))
        #resp_str = '<b>%s</b>有以下的老師開課：<br>%s' % (slot.get('title', courses[0].title), '<br>'.join(resp_list))
        resp_str = '<b>%s</b>有以下的老師開課：<br>%s' % (courses[0].title, '<br>'.join(resp_list))
    elif goal == 'title':
        courses = [c for c in courses if c.title != '']
        resp_list = list(np.unique([course.title for course in courses]))
        resp_str = '<b>%s</b>教授所開的課如下：<br>%s' % (courses[0].instructor, '<br>'.join(resp_list))
    elif goal == 'classroom':
        courses = [c for c in courses if c.classroom != '']
        resp_list = [course.classroom for course in courses]
        resp_str = '<b>%s</b>在<b>%s</b>上課。' % (courses[0].title, courses[0].classroom)
    elif goal == 'schedule':
        courses = [c for c in courses if c.schedule_str != '']
        resp_list = [course.schedule_str for course in courses]
        resp_str = '<b>%s</b>的上課時段在<b>星期%s節</b>。' % (courses[0].title, courses[0].schedule_str)

    return resp_list, resp_str


def understand(input):
    return {'tokens': [], 'labels': [], 'intent': [], 'slot': {}}
    tokens = [tok for tok in jieba.cut(input)]
    intent, tokens, labels = get_intent_slot(lu_model, tokens, word2idx, idx2label, idx2intent)

    print (tokens, labels, intent)
    d = {'tokens': tokens, 'labels': labels, 'intent': intent, 'slot': {}}
    for label, token in zip(labels, tokens):
        if label != 'O':
            d['slot'][label[2:]] = token

    return d
