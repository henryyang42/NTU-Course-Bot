"""
Created on May 18, 2016

@author: xiul, t-zalipt
"""
import os
import sys
sys.getdefaultencoding()  
import copy
from collections import defaultdict
from deep_dialog import dialog_config

import django
from crawler.const import base_url
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()
from django.db import models
from django.db.models import Q
from crawler.models import Course

def dict_to_KBquery(current_slots):
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


    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)

        Arguments:
        inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
        current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots

        Returns:
        filled_in_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
        """
        
        kb_results = self.available_results_from_kb(current_slots)
        if dialog_config.auto_suggest == 1:
            print ('Number of course in KB satisfying current constraints: ', len(kb_results))

        filled_in_slots = {}
        if 'taskcomplete' in inform_slots_to_be_filled.keys():
            filled_in_slots.update(current_slots['inform_slots'])
        
        for slot in inform_slots_to_be_filled.keys():
            """
            if slot == 'numberofpeople':
                if slot in current_slots['inform_slots'].keys():
                    filled_in_slots[slot] = current_slots['inform_slots'][slot]
                elif slot in inform_slots_to_be_filled.keys():
                    filled_in_slots[slot] = inform_slots_to_be_filled[slot]
                continue
            """
            if slot == 'taskcomplete':
                filled_in_slots[slot] = dialog_config.TICKET_AVAILABLE if len(kb_results)>0 else dialog_config.NO_VALUE_MATCH
                continue    
            if slot == 'closing': continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################
            values_dict = self.available_slot_values(slot, kb_results)

            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
            if len(values_counts) > 0:
                filled_in_slots[slot] = sorted(values_counts, key = lambda x: -x[1])[0][0]
            else:
                filled_in_slots[slot] = dialog_config.NO_VALUE_MATCH #"NO VALUE MATCHES SNAFU!!!"
           
        return filled_in_slots


    def available_slot_values(self, slot, kb_results):
        """ Return the set of values available for the slot based on the current constraints """
        
        slot_values = {}
        for movie_id in kb_results.keys():
            if slot in kb_results[movie_id].keys():
                slot_val = kb_results[movie_id][slot]
                if slot_val in slot_values.keys():
                    slot_values[slot_val] += 1
                else: slot_values[slot_val] = 1
        return slot_values

    def available_results_from_kb(self, current_slots):
        """ Return the available courses in the course_kb based on the current constraints """

        ret_result = {}
        query_slots = current_slots['inform_slots'].copy()
        query_slots.update(current_slots['proposed_slots'])
        kb_results = Course.objects.filter(dict_to_KBquery(query_slots))
        for i,course in enumerate(kb_results):
            ret_result[i] = {'serial_no':course.serial_no, 'title':course.title, 'instructor':course.instructor,'classroom':course.classroom,'schedule_str':course.schedule_str}
            

        ret_result = dict(ret_result)
        return ret_result
    
    def available_results_from_kb_for_slots(self, inform_slots):
        """ Return the count statistics for each constraint in inform_slots """
        
        kb_results = {key:0 for key in inform_slots.keys()}
        kb_results['matching_all_constraints'] = 0

        for key in inform_slots.keys():
            kb_results[key] = Course.objects.filter(dict_to_KBquery({key:inform_slots[key]})).count()
        kb_results['matching_all_constraints'] = Course.objects.filter(dict_to_KBquery(inform_slots)).count()

        return kb_results

    
    def database_results_for_agent(self, current_slots):
        """ A dictionary of the number of results matching each current constraint. The agent needs this to decide what to do next. """

        database_results ={} # { date:100, distanceconstraints:60, theater:30,  matching_all_constraints: 5}
        database_results = self.available_results_from_kb_for_slots(current_slots['inform_slots'])
        return database_results
    
    def suggest_slot_values(self, request_slots, current_slots):
        """ Return the suggest slot values """

        avail_kb_results = self.available_results_from_kb(current_slots)
        return_suggest_slot_vals = {}
        for slot in request_slots.keys():
            avail_values_dict = self.available_slot_values(slot, avail_kb_results)
            values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]
            
            if len(values_counts) > 0:
                return_suggest_slot_vals[slot] = []
                sorted_dict = sorted(values_counts, key = lambda x: -x[1])
                for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
            else:
                return_suggest_slot_vals[slot] = []
        
        return return_suggest_slot_vals