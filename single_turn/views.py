import json
from django.shortcuts import render
from django.http import HttpResponse

from crawler.models import *
from .request import *
# Create your views here.


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        print (user_input)
        try:
            d = json.loads(user_input)
            ans = query_course(goal=d['intent'], **d['slot'])
            print (ans)
        except:
            return HttpResponse("Server Error: %s" % user_input)

        return HttpResponse(' '.join(ans))

    return render(request, 'single_turn/single_turn.html', {})
