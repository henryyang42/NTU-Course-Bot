"""
Created on May 18, 2016

@author: xiul, t-zalipt
"""

import copy
import os
import sys
from collections import defaultdict
from dqn_agent import dialog_config
sys.path.append(os.getcwd())
sys.path.append(os.path.pardir)
from utils.query import *

class KBHelper:
    """ An assistant to fill in values for the agent (which knows about slots of values) """

    def __init__(self, course_dict):
        """ Constructor for a KBHelper """

        self.course_dict = course_dict
        self.cached_kb = defaultdict(list)
        self.cached_kb_slot = defaultdict(list)


    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)
        @Arguments:
            inform_slots_to_be_filled   --  Something that looks like {title: None,
                                            instructor: None} where title and
                                            instructor are slots that the agent needs
                                            filled
            current_slots               --  Contains a record of all filled slots in the
                                            conversation so far - for now,
                                            just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots
        @Returns:
            filled_in_slots             --  A dictionary of form like:
                                            {slot1: value1, slot2: value2}
                                            for each slot in inform_slots_to_be_filled
        """
        print('KB-Helper - fill_inform_slots -> inform_slots_to_be_filled: \n\t',
              inform_slots_to_be_filled, '\n')
        print('KB-Helper - fill_inform_slots -> current_slots: \n\t',
              current_slots, '\n')

        kb_results = self.available_results_from_kb(current_slots)
        if dialog_config.auto_suggest == 1:
            print('Number of courses in KB satisfying current constraints:\n\t', len(
                kb_results), '\n')

        filled_in_slots = {}
        # if 'taskcomplete' in inform_slots_to_be_filled.keys():
        #     filled_in_slots.update(current_slots['inform_slots'])

        for slot in inform_slots_to_be_filled.keys():
            # if slot == 'title':
            #     if slot in current_slots['inform_slots'].keys():
            #         filled_in_slots[slot] = current_slots['inform_slots'][slot]
            #     elif slot in inform_slots_to_be_filled.keys():
            #         filled_in_slots[slot] = inform_slots_to_be_filled[slot]
            #     continue

            # if slot == 'ticket' or slot == 'taskcomplete':
            #     filled_in_slots[slot] = dialog_config.TICKET_AVAILABLE if len(kb_results)>0 else dialog_config.NO_VALUE_MATCH
            #     continue

            # if slot == 'closing': continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################
            values_dict = self.available_slot_values(slot, kb_results)

            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
            if len(values_counts) > 0:
                filled_in_slots[slot] = sorted(values_counts, key = lambda x: -x[1])[0][0]
            else:
                filled_in_slots[slot] = dialog_config.NO_VALUE_MATCH # "NO VALUE MATCHES SNAFU!!!"

        return filled_in_slots

    def fill_choice_slots(self, choice_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)
        @Arguments:
            choice_slots_to_be_filled   --  A list of dictionaries of form looks like
                                            [{title: None}, {title: None}] where
                                            each element should the agent need
                                            filled
            current_slots               --  Contains a record of all filled slots in the
                                            conversation so far - for now,
                                            just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots
        @Returns:
            choice_slot                 --  A list of dictionaries of form like:
                                            [{slot1: value1}, {slot1: value2}]
                                            for each element in choice_slots_to_be_filled
        """
        print('KB-Helper - fill_inform_slots -> inform_slots_to_be_filled: \n\t',
              inform_slots_to_be_filled, '\n')
        print('KB-Helper - fill_inform_slots -> current_slots: \n\t',
              current_slots, '\n')

        kb_results = self.available_results_from_kb(current_slots)
        if dialog_config.auto_suggest == 1:
            print('Number of courses in KB satisfying current constraints:\n\t', len(
                kb_results), '\n')

        choice_slot = {}
        for slot in inform_slots_to_be_filled.keys():
            values_dict = self.available_slot_values(slot, kb_results)

            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
            if len(values_counts) > 0:
                choice_slot[slot] = sorted(values_counts, key=lambda x: -x[1])[0][0]
            else:
                # "NO VALUE MATCHES SNAFU!!!"
                choice_slot[slot] = dialog_config.NO_VALUE_MATCH

        return choice_slot

    def available_slot_values(self, slot, kb_results):
        """ Return the set of values available for the slot based on the current constraints """

        slot_values = {}
        for course_id in kb_results.keys():
            if slot in kb_results[course_id].keys():
                slot_val = kb_results[course_id][slot]
                if slot_val in slot_values.keys():
                    slot_values[slot_val] += 1
                else:
                    slot_values[slot_val] = 1
        return slot_values

    def available_results_from_kb(self, current_slots):
        """ Return the available courses in the course_kb based on the current constraints """

        ret_result = []
        current_slots = current_slots['inform_slots']
        constrain_keys = current_slots.keys()

        # constrain_keys = filter(lambda k : k != 'closing' , constrain_keys)
        constrain_keys = [k for k in constrain_keys if current_slots[k] != dialog_config.I_DO_NOT_CARE]

        query_idx_keys = frozenset(current_slots.items())
        cached_kb_ret = self.cached_kb[query_idx_keys]

        cached_kb_length = len(cached_kb_ret) if cached_kb_ret != None else -1
        if cached_kb_length > 0:
            return dict(cached_kb_ret)
        elif cached_kb_length == -1:
            return dict([])

        # kb_results = copy.deepcopy(self.course_dict)
        for id in self.course_dict.keys():
            kb_keys = self.course_dict[id].keys()
            if len(set(constrain_keys).union(set(kb_keys)) ^ (set(constrain_keys) ^ set(kb_keys))) == len(
                    constrain_keys):
                match = True
                for idx, k in enumerate(constrain_keys):
                    if str(current_slots[k]).lower() == str(self.course_dict[id][k]).lower():
                        continue
                    else:
                        match = False
                if match:
                    self.cached_kb[query_idx_keys].append((id, self.course_dict[id]))
                    ret_result.append((id, self.course_dict[id]))

        if len(ret_result) == 0:
            self.cached_kb[query_idx_keys] = None

        ret_result = dict(ret_result)
        return ret_result

    def available_results_from_kb_for_slots(self, inform_slots):
        """ Return the count statistics for each constraint in inform_slots """

        print("KB-Helper - available_results_from_kb_for_slots -> inform_slots:")
        for k, v in inform_slots.items():
            print('\t', "\"%s\":" % k, v)
        print()

        # initialize the database query results
        kb_results = {key: 0 for key in inform_slots.keys()}
        kb_results['matching_all_constraints'] = 0

        query_idx_keys = frozenset(inform_slots.items())
        cached_kb_slot_ret = self.cached_kb_slot[query_idx_keys]

        if len(cached_kb_slot_ret) > 0:
            return cached_kb_slot_ret[0]

        query_course_list = query_course(inform_slots)
        print("KB-Helper - available_results_from_kb_for_slots -> query_course_list:")
        for c in query_course_list:
            print('\t', "\"%s\":" % c)
        print()

        # here could be replaced with utils.query -> query_course
        for course_id in self.course_dict.keys():
            all_slots_match = 1
            for slot in inform_slots.keys():
                if inform_slots[slot] == dialog_config.I_DO_NOT_CARE:
                    continue

                dict_slot = 'schedule_str' if slot == 'when' else slot
                if dict_slot in self.course_dict[course_id].keys():
                    if inform_slots[slot].lower() == self.course_dict[course_id][dict_slot].lower():
                        kb_results[slot] += 1
                    else:
                        all_slots_match = 0
                else:
                    all_slots_match = 0
            kb_results['matching_all_constraints'] += all_slots_match

        self.cached_kb_slot[query_idx_keys].append(kb_results)
        return kb_results


    def database_results_for_agent(self, current_slots):
        """ A dictionary of the number of results matching each current constraint. The agent needs this to decide what to do next. """

        database_results = {}
        # {date: 100, distanceconstraints: 60, theater: 30,  matching_all_constraints: 5}
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
