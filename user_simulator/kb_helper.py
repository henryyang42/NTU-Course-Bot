import os
import django
from crawler.const import base_url
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()
from django.db import models
from django.db.models import Q
from crawler.models import Course

def dict_to_kbquery(current_slots):
    kb_query = Q()
    for slot in list(current_slots.keys()):
        kb_query &= Q(semester='105-2')
        if slot == 'title':
            kb_query &= Q(title=current_slots[slot])
        elif slot == 'instructor':
            kb_query &= Q(instructor=current_slots[slot])
        elif slot == 'classroom':
            kb_query &= Q(classroom=current_slots[slot])
        elif slot == 'schedule_str':
            kb_query &= Q(schedule_str=current_slots[slot])
        else:
            pass
    return kb_query




class KBHelper:
    """ An assistant to fill in values for the agent (which knows about slots of values) """
    
    def __init__(self):
        """ Constructor for a KBHelper """
        pass

    def query(self, constraint, slot):
        result = {slot:[]}
        kb_results = Course.objects.filter(dict_to_kbquery(constraint))
        
        result[slot] = list(kb_results.values_list(slot,flat=True))
        result['count'] = kb_results.count()
        return result

