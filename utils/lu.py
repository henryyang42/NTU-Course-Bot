import pickle

from .decorator import run_once
from .nlg import *
from .tagger import *
from DiaPol_rule.dia_pol import *
from LU_LSTM.lstm_predict import *
from django.conf import settings


@run_once
def multi_turn_lu_setup():
    global understand
    from multiturn_LU.test_MTLU import understand
    print('[Info] Multi-turn LU model loaded.')


def multi_turn_lu(user_id, sentence):
    multi_turn_lu_setup()
    status = understand(user_id, sentence)
    action = get_action_from_frame(status)
    # return status, action, agent2nl(action)
    return status, action, get_NL_from_action(action)


@run_once
def single_turn_lu_setup():
    global lu_model, idx2label, idx2intent, word2idx

    # load vocab
    obj = json.load(open('%s/LU_LSTM/re_seg.1K+log_extend_1000.vocab.json' % settings.BASE_DIR, "r"))
    idx2label = obj["slot_vocab"]
    idx2intent = obj["intent_vocab"]
    word2idx = {}
    for i, w in enumerate(obj["word_vocab"]):
        word2idx[w] = i

    # load model
    lu_model = load_model('%s/LU_LSTM/PY3--re_seg.1K+log_extend_1000--LSTM.model' % settings.BASE_DIR)
    print('[Info] Single-turn LU model loaded.')


def single_turn_lu(input):
    single_turn_lu_setup()
    tokens = cut(input)
    intent, tokens, labels = get_intent_slot(
        lu_model, tokens, word2idx, idx2label, idx2intent
    )

    print (tokens, labels, intent)
    d = {'tokens': tokens, 'labels': labels, 'intent': intent, 'slot': {}}
    for label, token in zip(labels, tokens):
        if label != 'O':
            d['slot'][label[2:]] = token

    return d
