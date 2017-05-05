# coding: utf-8
from access_django import *
import random
import numpy as np
from crawler.const import base_url
from django.template import Context, Template
from crawler.models import *


def formulate(**kwargs):
    kwlist = list(kwargs.items())
    len_kw = len(kwlist)
    state = [0] * (len_kw + 1)

    total = 1
    for _, l in kwlist:
        total *= len(l)

    print ('Formulating %d sentences...' % total)

    while True:
        d = {}
        for i, v in enumerate(state[:-1]):
            d[kwlist[i][0]] = kwlist[i][1][v]
        yield d

        state[0] += 1
        ind = 0

        while ind < len_kw and state[ind] == len(kwlist[ind][1]):
            state[ind] = 0
            state[ind + 1] += 1
            ind += 1

        if len_kw == ind:
            break


def BIO(sentence, context):
    inv_context = {v: k for k, v in context.items()}
    toks = sentence.split()
    tags = ['O'] * len(toks)
    for i, tok in enumerate(toks):
        tag = inv_context.get(tok, '')
        if tag in ['title', 'when', 'instructor']:
            tags[i] = 'B_%s' % tag

    return ' '.join(tags) + '\n'


pihua_start = ['', '請問', '告訴我', '我想知道', '幫我找', '幫我查', '幫我查一下', '查一下']
pihua_end = ['', '謝謝', '感謝']
where = ['在哪', '哪裡', '在哪裡', '哪間教室', '在哪間教室', '在什麼地方', '哪個館', '哪個系館', '在哪個系館', '在哪個系館哪間教室']
question = ['', '呢', '嗎', '你知道嗎']
when = ['', '今天', '星期一', '星期二', '星期三', '星期四', '星期五', '禮拜一', '禮拜二', '禮拜三', '禮拜四', '禮拜五']
course_query = ['有哪些課', '開哪些課', '有開哪些課', '教哪些課']
instructor_query = ['有哪些', '是哪個', '是哪位']
time_query = ['什麼時候', '在幾點', '在星期幾', '在禮拜幾', '幾點幾分']
teacher = ['', '老師', '教授']
titles = []
instructors = []

all_course = list(Course.objects.filter(semester='105-2'))

for course in all_course:
    titles.append(course.title)
    instructors.append(course.instructor)

titles = np.unique([x for x in titles if x and ' ' not in x])
instructors = np.unique([x for x in instructors if x and ' ' not in x])

print ('%d titles, %d instructors' % (len(titles), len(instructors)))


template_folder = 'request_template'
os.makedirs(template_folder, exist_ok=True)


with open('%s/classroom.txt' % template_folder, 'w') as f:
    tpl = Template('{{when}} {{title}} {{where}} 上課 {{question}} ?\n')

    random.shuffle(titles)
    for c in formulate(when=when, title=titles[:1000], where=where, question=question):
        context = Context(c)
        t = tpl.render(context)

        f.write(t)
        f.write(BIO(t, c))

    tpl = Template('{{when}} {{instructor}} 的 {{title}} {{where}} 上課 {{question}} ?\n')
    random.shuffle(all_course)
    for c in formulate(when=when, where=where, question=question):
        for course in all_course[:1000]:
            c['instructor'] = course.instructor
            c['title'] = course.title
            context = Context(c)
            t = tpl.render(context)

            f.write(t)
            f.write(BIO(t, c))


with open('%s/title.txt' % template_folder, 'w') as f:
    tpl = Template('{{pihua_start}} {{instructor}} {{teacher}} {{course_query}} {{question}} ?\n')

    for c in formulate(pihua_start=pihua_start, instructor=instructors, teacher=teacher, course_query=course_query, question=question):
        context = Context(c)
        t = tpl.render(context)

        f.write(t)
        f.write(BIO(t, c))


with open('%s/instructor.txt' % template_folder, 'w') as f:
    tpl = Template('{{pihua_start}} {{title}} {{instructor_query}} {{teacher}} ?\n')

    for c in formulate(pihua_start=pihua_start, title=titles, instructor_query=instructor_query, teacher=teacher):
        context = Context(c)
        t = tpl.render(context)

        f.write(t)
        f.write(BIO(t, c))

    tpl = Template('{{pihua_start}} {{title}} {{who}} {{pihua_end}} ?\n')

    for c in formulate(pihua_start=pihua_start, title=titles, pihua_end=pihua_end, who=['是誰開的', '誰開的', '誰上的', '是誰上的', '是誰開課', '是誰上課']):
        context = Context(c)
        t = tpl.render(context)

        f.write(t)
        f.write(BIO(t, c))


with open('%s/schedule.txt' % template_folder, 'w') as f:
    tpl = Template('{{pihua_start}} {{title}} 是 {{time_query}} 上課 {{pihua_end}} ?\n')

    for c in formulate(pihua_start=pihua_start, title=titles, time_query=time_query, pihua_end=pihua_end):
        context = Context(c)
        t = tpl.render(context)

        f.write(t)
        f.write(BIO(t, c))

    tpl = Template('{{pihua_start}} {{title}} {{time_query}} 上課 {{pihua_end}} ?\n')

    for c in formulate(pihua_start=pihua_start, title=titles, time_query=time_query, pihua_end=pihua_end):
        context = Context(c)
        t = tpl.render(context)

        f.write(t)
        f.write(BIO(t, c))
