import pickle

from .decorator import run_once
from .nlg import *
from .tagger import *
from DiaPol_rule.dia_pol import *
from LU_LSTM.lstm_predict import *
from django.conf import settings
from utils.query import *

@run_once
def multi_turn_lu_setup():
    global understand
    from multiturn_LU.test_MTLU import understand
    print('[Info] Multi-turn LU model loaded.')


def multi_turn_lu(user_id, sentence):
    multi_turn_lu_setup()
    status = understand(user_id, sentence)
    action = get_action_from_frame(status)
    return status, action, agent2nl(action)
    #return status, action, get_NL_from_action(action)


def multi_turn_lu2(user_id, sentence, reset=False):
    single_turn_lu_setup()
    with open('user_log.p', 'rb') as handle:
        user_log = pickle.load(handle)
    if reset:
        user_log[user_id] = {'request_slots': {}, 'inform_slots': {}}
        with open('user_log.p', 'wb') as handle:
            pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
    status = user_log.get(user_id, {'request_slots': {}, 'inform_slots': {}})
    d = single_turn_lu(sentence)
    if 'when' in d['slot']:
        d['slot']['schedule_str'] = d['slot']['when'][-1]
        d['slot'].pop('when')
    if not status['request_slots']:
        status['request_slots']['schedule_str' if d['intent'] == 'schedule' else d['intent']] = '?'
    for k, v in d['slot'].items():
        '''
        if k not in status['inform_slots']:
            status['inform_slots'][k] = v
        '''
        status['inform_slots'][k] = v # allow updating slots

    action = get_action_from_frame(status)
    # return status, action, agent2nl(action)
    if action['diaact'] in ['inform', 'closing']:
        user_log[user_id] = {'request_slots': {}, 'inform_slots': {}}
    else:
        user_log[user_id] = status
    with open('user_log.p', 'wb') as handle:
        pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return d, status, action, agent2nl(action)
    #return d, status, action, get_NL_from_action(action)


def set_status(user_id, status={'request_slots': {}, 'inform_slots': {}}):
    with open('user_log.p', 'rb') as handle:
        user_log = pickle.load(handle)
    user_log[user_id] = status
    with open('user_log.p', 'wb') as handle:
        pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)


def multi_turn_lu3(user_id, sentence, reset=False):
    single_turn_lu_setup_new()
    with open('user_log.p', 'rb') as handle:
        user_log = pickle.load(handle)
    if reset:
        set_status(user_id)
        return
    status = user_log.get(user_id, {'request_slots': {}, 'inform_slots': {}})
    d = single_turn_lu_new(sentence)
    if 'when' in d['slot']:
        d['slot']['schedule_str'] = d['slot']['when'][-1]
        d['slot'].pop('when')

    if d['intent'].startswith('request'):
        status['request_slots'][d['intent'][8:]] = '?'

    for k, v in d['slot'].items():
        if len(v) > 1 or k in ['schedule_str']:
            status['inform_slots'][k] = v
    # Retrieve reviews
    if d['intent'] == 'request_review':
        set_status(user_id)
        reviews = query_review(status['inform_slots'])
        review_resp = []
        if reviews.count() == 0:
            review_resp.append('並未搜尋到相關評價QQ')
        else:
            review_resp.append('幫您搜尋到%d筆相關評價：<br>' % reviews.count())
            for review in reviews:
                '<a target="_blank" href="https://www.ptt.cc/bbs/NTUCourse/%s.html">%s</a><br>' % (review.article_id, review.title)
        return d, status, {}, '\n'.join(review_resp)

    action = get_action_from_frame(status)

    if action['diaact'] in ['inform', 'closing']:
        user_log[user_id] = {'request_slots': {}, 'inform_slots': {}}
    else:
        user_log[user_id] = status
    with open('user_log.p', 'wb') as handle:
        pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return d, status, action, agent2nl(action)


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

    with open('user_log.p', 'wb') as handle:
        pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)

@run_once
def single_turn_lu_setup_new(): # load new LU models (output new intents)
    global lu_model, idx2label, idx2intent, word2idx

    # load vocab
    obj = json.load(open('%s/LU_LSTM/training_template0511.vocab.json' % settings.BASE_DIR, "r"))
    idx2label = obj["slot_vocab"]
    idx2intent = obj["intent_vocab"]
    word2idx = {}
    for i, w in enumerate(obj["word_vocab"]):
        word2idx[w] = i

    # load model
    lu_model = load_model('%s/LU_LSTM/PY3--training_template0511--LSTM.model' % settings.BASE_DIR)
    print('[Info] Single-turn LU model loaded.')

    with open('user_log.p', 'wb') as handle:
        pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)



def single_turn_lu(sentence):
    single_turn_lu_setup()
    tokens = cut(sentence)
    intent, tokens, labels = get_intent_slot(
        lu_model, tokens, word2idx, idx2label, idx2intent
    )

    print (tokens, labels, intent)
    d = {'tokens': tokens, 'labels': labels, 'intent': intent, 'slot': {}}
    for label, token in zip(labels, tokens):
        if label != 'O':
            d['slot'][label[2:]] = token
    #FIXME handle multiple B_xx for same slot (rule-based decision?)
    return d

def single_turn_lu_new(sentence):
    single_turn_lu_setup_new()
    tokens = cut(sentence)
    intent, tokens, labels = get_intent_slot(
        lu_model, tokens, word2idx, idx2label, idx2intent
    )

    print (tokens, labels, intent)
    d = {'tokens': tokens, 'labels': labels, 'intent': intent, 'slot': {}}
    # select heuristically from multiple B_xx for same slot
    slot_value_list = {}
    for label, token in zip(labels, tokens):
        if label != 'O':
            slot, value = label[2:], token
            if slot not in slot_value_list:
                slot_value_list[slot] = []
            slot_value_list[slot].append( value )
    for slot in slot_value_list: # Comparison rule: (1)longer first (2)left first
        max_n_char = 0
        best_value = None
        for value in slot_value_list[slot]:
            n_char = len(list(value))
            if n_char > max_n_char:
                max_n_char = n_char
                best_value = value
        d['slot'][slot] = best_value
    return d
