import json
import random

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render

from crawler.models import *
from utils.query import *


def toggle_rating(request, id):
    d_log = get_object_or_404(DialogueLog, id=id)
    d_log.toggle_rating()
    return HttpResponse("OK")


def set_rating(request, group_id, rating):
    d_group = get_object_or_404(DialogueLogGroup, group_id=group_id)
    d_group.reward = rating
    d_group.save()
    return HttpResponse("OK")


def get_course(request, serial_no):
    course = get_object_or_404(Course, semester='105-2', serial_no=serial_no)
    return JsonResponse({k: v for k, v in course.__dict__.items() if not k.startswith('_')})


def tag_data(request):
    if request.method == 'POST':
        d_log = get_object_or_404(DialogueLog, id=request.POST['id'])
        d_log.tagged_data = request.POST['tagged_data']
        d_log.tagged = True
        d_log.save()
        return HttpResponse("OK")
    untagged = DialogueLog.objects.filter(tagged=False)
    tagged = DialogueLog.objects.filter(tagged=True)
    sementic = {}
    for _ in range(100):
        dialogue = untagged[random.randint(0, untagged.count() - 1)]
        try:
            debug = json.loads(dialogue.debug.replace("'", '"'))
            sementic = debug['sementic']
            break
        except:
            pass
    return render(request, 'web_api/tag_data.html',
                  {'untagged_count': untagged.count(),
                   'tagged_count': tagged.count(),
                   'dialogue': dialogue,
                   'sementic': json.dumps(sementic, ensure_ascii=False)
                   })


def delete_dialogue(request, id):
    d_log = get_object_or_404(DialogueLog, id=id)
    d_log.delete()
    return HttpResponse("OK")


def download_log(request):
    tagged_data = []
    for data in DialogueLog.objects.filter(tagged=True):
        try:
            tagged_data.append(json.loads(data.tagged_data))
        except:
            pass
    response = HttpResponse(json.dumps(tagged_data, ensure_ascii=False), content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename="dialogue_log.json"'
    return response
