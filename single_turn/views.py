import json
import logging
from django.shortcuts import render
from django.http import HttpResponse

from crawler.models import *
from .request import *

# Create your views here.
logger = logging.getLogger(__name__)


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        try:
            d = understand(user_input)
            d['resp_list'], d['resp_str'] = query_course(d['intent'], d['slot'])
            logger.debug('%s -> %s\n%s' % (user_input, d['resp_str'], str(d)))
            return HttpResponse("%s<br><br>%s" % (d['resp_str'], str(d)))
        except Exception as e:
            print(e)

        return HttpResponse("我壞掉惹QQ")

    return render(request, 'single_turn/single_turn.html', {})
