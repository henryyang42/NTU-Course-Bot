import json
import logging
import traceback
import sys
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse

from crawler.models import *
from .request import *

# Create your views here.
logger = logging.getLogger(__name__)


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        resp = {'resp_str': '我壞掉惹QQ'}
        try:
            resp = understand(user_input)
            resp['resp_list'], resp['resp_str'] = query_course(resp['intent'], resp['slot'])
            logger.debug('%s -> %s\n%s' % (user_input, resp['resp_str'], str(resp)))
            d_log = DialogueLog.objects.create(utterance=user_input, reply=resp['resp_str'], debug=resp)
            resp['id'] = d_log.id
        except Exception:
            traceback.print_exc(file=sys.stdout)
            DialogueLog.objects.create(utterance=user_input, debug=traceback.format_exc())

        return HttpResponse(json.dumps(resp), content_type="application/json")

    return render(request, 'single_turn/single_turn.html', {})


def toggle_rating(request, id):
    d_log = get_object_or_404(DialogueLog, id=id)
    d_log.toggle_rating()
    return HttpResponse("OK")
