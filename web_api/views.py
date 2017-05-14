from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse

from crawler.models import *
from utils.query import *


def toggle_rating(request, id):
    d_log = get_object_or_404(DialogueLog, id=id)
    d_log.toggle_rating()
    return HttpResponse("OK")


def get_course(request, serial_no):
    course = get_object_or_404(Course, semester='105-2', serial_no=serial_no)
    return JsonResponse({k: v for k, v in course.__dict__.items() if not k.startswith('_')})

