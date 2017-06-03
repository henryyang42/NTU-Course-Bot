import sys
import os
import random

def get_rand_action():
    # required fields
    sys_act = {} 
    sys_act["diaact"] = random.choice(["inform", "request", "multiple_choice", "confirm", "closing"])
    sys_act["inform_slots"] = {}
    sys_act["request_slots"] = {}
    
    # decide action by # courses satisfying the constraints
    unique_found = False
    if course_ct == 0:  # no course satisfying all constraints
        wrong_slot = None
        # remove the constraints one by one
        for slot in ["title", "instructor", "schedule_str"]:
            if slot not in dia_state["request_slots"] and slot in dia_state["inform_slots"]:
                tmp_inform_slots = dia_state["inform_slots"].copy()
                del tmp_inform_slots[slot]
                #print (tmp_inform_slots)
                tmp_cnt = query_course(tmp_inform_slots).count()
                print ("[INFO] try removing slot %s, # courses = %d" % (slot, tmp_cnt))
                if tmp_cnt > 0:
                    wrong_slot = slot
                    break
        if wrong_slot is not None:
            sys_act["diaact"] = "confirm"
            sys_act["inform_slots"][wrong_slot] = dia_state["inform_slots"][wrong_slot]#TODO
        else:
            sys_act["diaact"] = "closing" # not satisfiable

    elif course_ct == 1:
        unique_found = True

    else: # [ len(courses) > 1 ] `request` / `multiple_choice`
        req_slot = None
        choice_set = None
        req_max_n = 0 # only consider the slots that can be requested
        #TODO refine a set of slots that the system can request
        for slot in ["title", "instructor", "designated_for", "schedule_str"]:# ordered by priority
            # max # different values --> largest diversity
            values_set = set([c[slot] for c in courses if len(c[slot])>0])
            n_values = len(values_set)
            print ("[INFO] slot %s, # values = %d" % (slot, n_values))

            if n_values > req_max_n:
                if n_values > MAX_N_CHOICE: # not taking `multiple_choice` action
                    # don't ask users something they are askin
                    if slot in dia_state["request_slots"]:
                        continue
                # don't ask users something already known
                #if slot in dia_state["inform_slots"] and slot != "schedule_str": # "schedule_str" could be incomplete
                if slot in dia_state["inform_slots"]:
                    continue

                req_max_n = n_values
                req_slot = slot
                choice_set = values_set

        if req_max_n <= 1: # only one course satisfy the constraints
            unique_found = True
        elif req_max_n <= MAX_N_CHOICE: # no more than 5 values => `multiple_choice`
            sys_act["diaact"] = "multiple_choice"
            sys_act["choice"] = [{req_slot:v} for v in choice_set] # pass list to user
        elif req_slot is not None: # `request`
            sys_act["diaact"] = "request"
            sys_act["request_slots"][req_slot] = "?"

    if unique_found: # find the unique course
        course = courses[0]
        sys_act["diaact"] = "inform"
        inform_slots = {}
        for slot in dia_state["request_slots"]:
            if slot in course:
                inform_slots[slot] = course[slot]
        
        ### slots that must be informed ###
        inform_slots["serial_no"] = course["serial_no"] # must provide serial_no to complete the task
        inform_slots["title"] = course["title"] # return course name the ensure the correct course is found
        inform_slots["instructor"] = course["instructor"] # for querying review
        ### ### ###
        
        sys_act["inform_slots"] = inform_slots

    return sys_act




if __name__ == '__main__':
    while True:
        sys_act = get_rand_action()
        print (json.dumps(sys_act))
        input()
