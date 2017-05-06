# coding: utf-8
from access_django import *
from tabulate import tabulate
from crawler.models import *

rf = ["user", "review", "id", "schedule"]
cf = Course._meta.get_fields()
course_attrs = [f.name for f in cf if f.name not in rf]
course_attrs_cap = [a.capitalize() for a in course_attrs]

ml = Course.objects.filter(title='機器學習').values()[0]
ms = Course.objects.filter(title='音樂社會學').values()[0]
ky = Course.objects.filter(title='金庸武俠小說').values()[0]
course_table = [[] for _ in range(3)]
course_table[0] = [ml[k] for k in course_attrs]
course_table[1] = [ms[k] for k in course_attrs]
course_table[2] = [ky[k] for k in course_attrs]

with open('./crawler/md-based_course-table_example.html', 'w') as f:
    f.write(tabulate(course_table, course_attrs_cap, numalign='center', stralign='center', tablefmt='html'))

    rf = ["user", "review", "id", "schedule", "course"]
cf = Review._meta.get_fields()
review_attrs = [f.name for f in cf if f.name not in rf]
review_attrs_cap = [a.capitalize() for a in review_attrs]

th = Review.objects.filter(title='[評價] 105-1 臺灣史一 李文良').values()[0]
lu = Review.objects.filter(title='[評價] 104-2 黃奕珍 陸游詩').values()[0]
ch = Review.objects.filter(title='[評價] 104-2 李隆獻 大一國文下 ').values()[0]
review_table = [[] for _ in range(3)]
review_table[0] = [th[k] for k in review_attrs]
review_table[1] = [lu[k] for k in review_attrs]
review_table[2] = [ch[k] for k in review_attrs]

with open('./crawler/md-based_review-table_example.html', 'w') as f:
    f.write(tabulate(review_table, review_attrs_cap, numalign='center', stralign='center', tablefmt='html'))
