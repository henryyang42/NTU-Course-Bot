# coding: utf-8
from access_django import *
import random
import numpy as np
from crawler.const import base_url
from django.template import Context, Template
from crawler.models import *

# Templates
# possible_slots = [
# 'title',
# 'when',
# 'instructor',
# 'classroom',
# 'designated_for',
# 'required_elective',
# 'sel_method'
# ]

templates = {
    'request_title': [
        Template('有開哪些課'),
        Template('有哪些課'),
        Template('{{instructor}}在{{when}}有哪些課')
    ],
    'request_instructor': [
        Template('老師是誰'),
        Template('老師是哪位'),
        Template('開{{title}}的老師是哪位'),
    ],
    'request_schedule_str': [
        Template('什麼時候的課'),
        Template('上課時間在什麼時候')
    ],
    'request_classroom': [
        Template('在哪裡上課'),
        Template('教室在哪')
    ],
    'request_review': [
        Template(''),
    ],
    'request_designated_for': [
        Template(''),
    ],
    'request_required_elective': [
        Template(''),
    ],
    'request_sel_method': [
        Template(''),
    ],
    'inform': [
        Template(''),
    ],
    'other': [
        Template(''),
    ]
}
