import json
from django.shortcuts import render
from django.http import HttpResponse

from crawler.models import *
from .request import *

# Create your views here.


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        try:
            d = understand(user_input)
            d['resp_list'], d['resp_str'] = query_course(d['intent'], d['slot'])
        except Exception as e:
            print(e)

        return HttpResponse("%s<br><br>%s" % (d['resp_str'], str(d)))

    return render(request, 'single_turn/single_turn.html', {})
