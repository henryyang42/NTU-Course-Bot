import json
import logging
import traceback
import sys
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse

from crawler.models import *
from utils.lu import single_turn_lu, multi_turn_lu2
from utils.query import query_course
# Create your views here.
logger = logging.getLogger(__name__)


def multi_turn(request):
    if request.method == 'POST':
        resp = {'resp_str': '我壞掉惹QQ'}
        user_input = request.POST['input']
        uid = request.COOKIES['csrftoken']
        if user_input == 'reset':
            user_input = '我想選課'
            resp['resp_str'] = '已重設'
            multi_turn_lu2(uid, user_input, reset=True)
            return HttpResponse(json.dumps(resp), content_type="application/json")
        try:
            resp['status'], resp['action'], resp['resp_str'] = multi_turn_lu2(uid, user_input)
            # resp['resp_list'] = str([c.title for c in query_course(resp['sementic']['slot'])][:10])
            logger.debug('%s -> %s\n%s' % (user_input, resp['resp_str'], str(resp)))
            d_log = DialogueLog.objects.create(
                utterance=user_input, reply=resp['resp_str'], debug=resp)
            resp['id'] = d_log.id
        except Exception:
            traceback.print_exc(file=sys.stdout)
            DialogueLog.objects.create(
                utterance=user_input, debug=traceback.format_exc())

        return HttpResponse(json.dumps(resp), content_type="application/json")

    return render(request, 'single_turn/single_turn.html', {})