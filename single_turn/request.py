from crawler.models import *
import numpy as np


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
    courses = Course.objects.filter(**kwargs).filter(semester='105-2')
    for course in courses:
        eval('ans.append(course.%s)' % goal)

    return np.unique(ans)
