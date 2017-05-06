import random
from django.template import Context, Template
from .query import query_course

# Fillers
where = ['在哪', '哪裡', '在哪裡', '哪間教室', '在哪間教室',
         '在什麼地方', '哪個館', '哪個系館', '在哪個系館', '在哪個系館哪間教室']
when = ['今天', '星期一', '星期二', '星期三', '星期四',
        '星期五', '禮拜一', '禮拜二', '禮拜三', '禮拜四', '禮拜五']
course_query = ['有哪些課', '開哪些課', '有開哪些課', '教哪些課']
instructor_query = ['有哪些', '是哪個', '是哪位']
time_query = ['什麼時候', '在幾點', '在星期幾', '在禮拜幾', '幾點幾分']

# Templates
request_tpl = {
    'title': [
        Template('請列出課程名稱'),
        Template('什麼課')
    ],
    'instructor': [
        Template('老師的名字'),
        Template('老師是誰')
    ],
    'schedule_str': [
        Template('這堂課在星期幾上課?'),
        Template('上課時間在什麼時候')
    ],
    'classroom': [
        Template('在哪裡上課'),
        Template('教室在哪')
    ]
}

inform_tpl = {
    'title': [
        Template('{{title}}'),
        Template('課名是{{title}}'),
        Template('課程名稱為{{title}}')
    ],
    'instructor': [
        Template('{{instructor}}'),
        Template('教學老師是{{instructor}}'),
        Template('教學老師是{{instructor}}'),
        Template('{{instructor}}上的課'),
        Template('是{{instructor}}教授')
    ],
    'schedule_str': [
        Template('{{when}}'),
        Template('上課時間是{{when}}'),
        Template('我想上{{when}}的課')
    ],
    'classroom': [
        Template('{{classroom}}'),
        Template('上課教室是{{classroom}}'),
        Template('在{{classroom}}上課'),
        Template('在{{classroom}}')
    ]
}


def sem2nl(sem_in):
    """Convert sementic to NL using template based NLG.
    """
    if sem_in['diaact'] == 'request':
        attr = next(iter(sem_in['request_slots']))
        tpl = random.choice(request_tpl[attr])
        return tpl.render(Context(sem_in['request_slots']))
    elif sem_in['diaact'] == 'inform':
        attr = next(iter(sem_in['inform_slots']))
        tpl = random.choice(inform_tpl[attr])
        return tpl.render(Context(sem_in['inform_slots']))
    else:
        return '謝謝！'


def agent2nl(sem_in):
    if sem_in['diaact'] == 'request':
        attr = next(iter(sem_in['request_slots']))
        tpl = random.choice(request_tpl[attr])
        return tpl.render(Context(sem_in['request_slots']))
    elif sem_in['diaact'] == 'inform':
        course = query_course(sem_in['inform_slots']).values()
        if course.count() > 0:
            response = "流水號{serial_no} : {instructor}教授的{title}, 上課時間為{schedule_str}, 在{classroom}上課".format_map(
                course[0])
        else:
            response = "沒有找到相關課程"
    else:
        return '謝謝！'
