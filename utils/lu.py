import pickle

from .decorator import run_once
from .nlg import *

from DiaPol_rule.dia_pol import *


@run_once
def multi_turn_lu_setup():
    global understand
    from multiturn_LU.test_MTLU import understand


def multi_turn_lu(user_id, sentence):
    multi_turn_lu_setup()
    status = understand(user_id, sentence)
    action = get_action_from_frame(status)
    #return status, action, agent2nl(action)
    return status, action, get_NL_from_action(action)

