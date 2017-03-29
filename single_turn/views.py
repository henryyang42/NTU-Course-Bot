from django.shortcuts import render
from django.http import HttpResponse

from crawler.models import *
# Create your views here.


def single_turn(request):
    if request.method == 'POST':
        user_input = request.POST['input']
        print (user_input)
        courses = Course.objects.filter(title__contains=user_input)
        return HttpResponse(courses)
    return render(request, 'single_turn/single_turn.html', {})
