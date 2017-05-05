from django import forms
from django.shortcuts import render
from django.http import JsonResponse

from .demo import usim_initial, usim_request

# Create your views here.


class DialogAct(forms.Form):
    action = forms.ChoiceField(
        choices=(
            ('request', 'request'),
            ('inform', 'inform'),
            ('thanks', 'thanks'),
            ('closing', 'closing')
        )
    )
    """
    serial = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': '流水號'}),
        required=False)
    """
    title = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': '課程名稱'}),
        required=False)
    instructor = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': '教授'}),
        required=False)
    classroom = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': '教室'}),
        required=False)
    schedule = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': '時間'}),
        required=False)


def user_simulator(request):
    DiaactForm = DialogAct()
    if request.method == 'POST':
        DiaactForm = DialogAct(request.POST)
        if DiaactForm.is_valid():
            slots = {
                # 'serial_no':DiaactForm.cleaned_data['serial'],
                'title': DiaactForm.cleaned_data['title'],
                'instructor': DiaactForm.cleaned_data['instructor'],
                'classroom': DiaactForm.cleaned_data['classroom'],
                'schedule_str': DiaactForm.cleaned_data['schedule'],
            }

            if DiaactForm.cleaned_data['action'] == 'request':
                dialogue = {
                    'diaact': DiaactForm.cleaned_data['action'],
                    'request_slots': slots,
                    'inform_slots': {},
                }
            elif DiaactForm.cleaned_data['action'] == 'inform':
                dialogue = {
                    'diaact': DiaactForm.cleaned_data['action'],
                    'request_slots': {},
                    'inform_slots': slots,
                }
            elif DiaactForm.cleaned_data['action'] == 'thanks':
                dialogue = {
                    'diaact': 'thanks',
                    'request_slots': {},
                    'inform_slots': {},
                }
            elif DiaactForm.cleaned_data['action'] == 'closing':
                dialogue = {
                    'diaact': 'closing',
                    'request_slots': {},
                    'inform_slots': {},
                }
            else:
                pass

            dialogue['request_slots'] = {
                k: v for k, v in dialogue['request_slots'].items() if v}
            dialogue['inform_slots'] = {
                k: v for k, v in dialogue['inform_slots'].items() if v}


            response = JsonResponse(usim_request(dialogue), safe=False)

            return response

    goal, action, num, suggest = usim_initial()
    return render(request, 'user_simulator/usersim.html', {
        'diaactform': DiaactForm,
        'user_goal': goal,
        'user_action': action,
        'num': num,
        'suggest': suggest,
    })
