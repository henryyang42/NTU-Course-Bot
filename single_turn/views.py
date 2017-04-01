import json
from django.shortcuts import render
from django.http import HttpResponse

from crawler.models import *
from .request import *

# Create your views here.


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        understand(user_input)
        try:
            ans = query_course(goal=d['intent'], **d['slot'])
        except:
            ans = Course.objects.filter(semester='105-2', title__contains=user_input) | Course.objects.filter(semester='105-2', instructor__contains=user_input)
            if ans:
                return HttpResponse('<br>'.join([str(x) for x in ans]))
            else:
                return HttpResponse("Server Error: %s" % user_input)

        return HttpResponse('<br>'.join(ans))

    return render(request, 'single_turn/single_turn.html', {})
