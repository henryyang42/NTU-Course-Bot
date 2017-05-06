from crawler.models import *
import numpy as np
from django.db.models import Q


def expand_title(title):
    q = Q()
    for c in title:
        q &= Q(title__contains=c)
    return q


def query_course(constraints):
    """Return list of Course objects
    """
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
    courses = Course.objects.filter(expand_title(constraints.get('title', ''))).filter(**query_term).filter(semester='105-2')

    return courses

