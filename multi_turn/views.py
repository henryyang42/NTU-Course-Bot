import json
import logging
import traceback
import sys
import random
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse

from crawler.models import *
from utils.lu import multi_turn_lu
# Create your views here.
logger = logging.getLogger(__name__)

uid = 123456


def multi_turn(request):
    if request.method == 'POST':
        resp = {'resp_str': '我壞掉惹QQ'}
        user_input = request.POST['input']
        global uid
        if user_input == 'reset':
            user_input = '我想選課'
            uid = random.randint(0, 2147483647)
            resp['resp_str'] = '已重設'
            return HttpResponse(json.dumps(resp), content_type="application/json")
        print(uid)
        try:
            resp['status'], resp['action'], resp['resp_str'] = multi_turn_lu(uid, user_input)
            # resp['resp_list'], resp['resp_str'] = query_course(resp['intent'], resp['slot'])
            logger.debug('%s -> %s\n%s' % (user_input, resp['resp_str'], str(resp)))
            d_log = DialogueLog.objects.create(utterance=user_input, reply=resp['resp_str'], debug=resp)
            resp['id'] = d_log.id
        except Exception:
            traceback.print_exc(file=sys.stdout)
            DialogueLog.objects.create(utterance=user_input, debug=traceback.format_exc())

        return HttpResponse(json.dumps(resp), content_type="application/json")

    return render(request, 'single_turn/single_turn.html', {})
