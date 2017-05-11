'''
Created on May 11, 2017

@author: haley
'''

sys_request_slots = ['title', 'instructor',
                     'schedule_str']  # ordered by priority

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
TICKET_AVAILABLE = 'Ticket Available'

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
run_mode = 0
auto_suggest = 0

#########################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
#########################################################################
feasible_actions = [
    #####################################################################
    #   request slot action
    #####################################################################
    {'diaact': "request", 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   confirm slot action
    #####################################################################
    {'diaact': "confirm", 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   multiple_choice action
    #####################################################################
    {'diaact': "multiple_choice", 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   inform action
    #####################################################################
    {'diaact': "inform", 'inform_slots': {}, 'request_slots': {}},

    #####################################################################
    #   close action
    #####################################################################
    {'diaact': "closing", 'inform_slots': {}, 'request_slots': {}},
]

#########################################################################
#   Adding the inform actions
#########################################################################
for slot in sys_inform_slots:
    feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})

#########################################################################
#   Adding the request actions
#########################################################################
for slot in sys_request_slots:
    feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})
