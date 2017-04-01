import json
from django.shortcuts import render
from django.http import HttpResponse

from crawler.models import *
from .request import *

# Create your views here.


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        d = understand(user_input)
        try:
            ans = query_course(goal=d['intent'], **d['slot'])
        except:
            pass

        return HttpResponse("%s<br><br>%s" % (str(ans), str(d)))

    return render(request, 'single_turn/single_turn.html', {})
