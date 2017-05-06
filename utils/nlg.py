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
        Template('有開哪些課'),
        Template('有哪些課')
    ],
    'instructor': [
        Template('老師是誰'),
        Template('老師是哪位')
    ],
    'schedule_str': [
        Template('什麼時候的課'),
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
        Template('是{{title}}'),
    ],
    'instructor': [
        Template('{{instructor}}'),
        Template('是{{instructor}}老師'),
        Template('是{{instructor}}上的課'),
        Template('是{{instructor}}教授')
    ],
    'schedule_str': [
        Template('{{schedule_str}}'),
        Template('{{schedule_str}}開的課'),
        Template('{{schedule_str}}的課')
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
    if 'schedule_str' in sem_in['request_slots']:
        sem_in['request_slots']['schedule_str'] = sem_in['request_slots']['schedule_str'][1]
    if 'schedule_str' in sem_in['inform_slots']:
        sem_in['inform_slots']['schedule_str'] = sem_in['inform_slots']['schedule_str'][1]

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


def agent2nl(sys_act):
    if sys_act["diaact"] == "closing" and len(sys_act["inform_slots"]) == 0:
        return "不好意思，沒有找到符合條件的課程。"

    res_list = []
    # reponse in a pre-defined order
    for slot in ["serial_no", "title", "instructor", "classroom", "schedule_str"]:
        if slot in sys_act["inform_slots"]:
            if slot == "serial_no":
                res_str = "流水號%s。"
            elif slot == "title":
                res_str = "課名是%s。"
            elif slot == "instructor":
                res_str = "授課教師是%s。"
            elif slot == "classroom":
                res_str = "在%s上課。"
            elif slot == "schedule_str":
                res_str = "%s上課。"
            res_str = res_str % sys_act["inform_slots"][slot]
            res_list.append(res_str)

    # reponse in a pre-defined order
    for slot in ["title", "instructor", "classroom", "schedule_str"]:
        if slot in sys_act["request_slots"]:
            if slot == "title":
                res_str = "請問要找哪門課?"
            elif slot == "instructor":
                res_str = "請問是哪位老師開的?"
            elif slot == "classroom":
                res_str = "請問是在哪上課的?"
            elif slot == "schedule_str":
                res_str = "請問是哪個時間上課的?"
            res_list.append(res_str)

    return "".join(res_list)

'''
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
        response = '謝謝！'

    return response
'''
