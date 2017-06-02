'''
Created on May 11, 2017

@author: haley
'''
import itertools


sys_request_slots = ['serial_no', 'title', 'instructor',
                     'classroom', 'schedule_str', 'designated_for',
                     'required_elective', 'sel_method']

sys_inform_slots = ['serial_no', 'title', 'instructor',
                    'classroom', 'schedule_str', 'designated_for',
                    'required_elective', 'sel_method']

start_dia_acts = {
    'request': ['serial_no', 'title', 'instructor',
                'classroom', 'schedule_str', 'designated_for',
                'required_elective', 'sel_method']
}

#########################################################################
# Dialog status
#########################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
PENALTY_DIALOG = -10 # Penalty. (agent request the constraints had been informed before)
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 100
FAILURE_REWARD = -100
PER_TURN_REWARD = -1

#########################################################################
#  Special Slot Values
#########################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
COURSE_AVAILABLE = 'Course Available'

#########################################################################
#  Constraint Check
#########################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

#########################################################################
#  NLG Beam Search
#########################################################################
nlg_beam_size = 10

#########################################################################
#  run_mode:
#           0 for default NL
#           1 for dia_act
#           2 for both
#########################################################################
run_mode = 3
auto_suggest = 1

#########################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
#########################################################################
feasible_actions = [
    #####################################################################
    #   request slot action
    #####################################################################
    {'diaact': "request", 'choice': [], 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   confirm slot action
    #####################################################################
    {'diaact': "confirm", 'choice': [], 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   multiple_choice action
    #####################################################################
    {'diaact': "multiple_choice", 'choice': [], 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   inform action
    #####################################################################
    {'diaact': "inform", 'choice': [], 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   close action
    #####################################################################
    {'diaact': "closing", 'choice': [], 'inform_slots': {}, 'request_slots': {}},
]

#########################################################################
#   Adding all the possible inform actions
#########################################################################
for r in range(1, len(sys_inform_slots) + 1):
    for c in itertools.combinations(sys_inform_slots, r):
        feasible_actions.append({'diaact': 'inform',
                                 'inform_slots': {slot: "PLACEHOLDER" for slot in c},
                                 'request_slots': {},
                                 'choice': []})


#########################################################################
#   Adding all the possible request actions (including 'choice')
#########################################################################
for r in range(1, len(sys_request_slots) + 1):
    for c in itertools.combinations(sys_request_slots, r):
        feasible_actions.append({'diaact': 'request',
                                 'inform_slots': {},
                                 'request_slots': {slot: "UNK" for slot in c},
                                 'choice': []})


#########################################################################
#   Adding all the possible confirm actions
#########################################################################
for r in range(1, len(sys_inform_slots) + 1):
    for c in itertools.combinations(sys_inform_slots, r):
        feasible_actions.append({'diaact': 'confirm',
                                 'inform_slots': {slot: "PLACEHOLDER" for slot in c},
                                 'request_slots': {},
                                 'choice': []})
