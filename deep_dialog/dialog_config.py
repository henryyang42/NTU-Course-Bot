'''
Created on May 17, 2016

@author: xiul, t-zalipt
'''

sys_request_slots = ['title', 'instructor', 'classroom', 'schedule_str']
#sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']
sys_inform_slots = ['serial_no', 'title', 'instructor', 'classroom', 'schedule_str']
#sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']

start_dia_acts = {
    #'greeting':[],
    #'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
    'request': ['title', 'instructor', 'classroom', 'schedule_str']
}

################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "隨便, 都可以啦~"
NO_VALUE_MATCH = "沒有找到 >_<"
TICKET_AVAILABLE = '找到相關課程'

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 3
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [
    ############################################################################
    #   greeting actions
    ############################################################################
    #{'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_question actions
    ############################################################################
    {'diaact': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    #   confirm_answer actions
    ############################################################################
    {'diaact': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    #   deny actions
    ############################################################################
    {'diaact': "deny", 'inform_slots': {}, 'request_slots': {}},
]
############################################################################
#   Adding the inform actions
############################################################################
for slot in sys_inform_slots:
    feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})

############################################################################
#   Adding the request actions
############################################################################
for slot in sys_request_slots:
    feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})
