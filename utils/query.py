from crawler.models import *
import numpy as np
from django.db.models import Q
from .decorator import run_once


def expand_title(title):
    q = Q()
    for c in title:
        q &= Q(title__contains=c)
    return q


def course_attr_blank(c):
    return not (c.title and c.instructor and c.schedule_str and c.classroom)


def course_equal(c1, c2):
    return c1.title == c2.title and c1.instructor == c2.instructor and c2.schedule_str[0] == c1.schedule_str[0]


@run_once
def init_unique_course():
    # Remove redundant courses (same attr but different dept.)
    global unique_courses
    unique_courses = Course.objects.none()
    for c1 in Course.objects.filter(semester='105-2'):
        if not course_attr_blank(c1) and not any([course_equal(c1, c2) for c2 in unique_courses]):
            unique_courses._result_cache.append(c1)

    unique_courses = Course.objects.filter(id__in=[course.id for course in unique_courses])
    print('[Info] %d unique courses initiated.' % unique_courses.count())


def query_course(constraints):
    """Return list of Course objects
    """
    init_unique_course()
    # Transform slot to query terms.
    query_term = {}
    for k, v in constraints.items():
        if k == 'when':
            query_term['schedule_str__contains'] = v[-1]
        elif k == 'title':
            pass
        else:
            query_term[k + '__contains'] = v

    # Generate corresponding response to each intent.
    courses = unique_courses.filter(**query_term).filter(expand_title(constraints.get('title', '')))

    return courses

