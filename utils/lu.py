import json

from .decorator import run_once
from .nlg import *
from .tagger import *
from DiaPol_rule.dia_pol import *
from LU_LSTM.lstm_predict import *
from django.conf import settings
from utils.query import *
from utils.misc import *
from dqn.agent_dqn import *
from dqn.dialog_config import *
from datetime import datetime


def DST_update(old_state, sem_frame):
    state = old_state.copy()
    
    # process 'when'
    if 'when' in sem_frame['slot']:
        sem_frame['slot']['schedule_str'] = sem_frame['slot']['when'][-1]
        sem_frame['slot'].pop('when')

    # handle inform_unknown
    if sem_frame['intent'] == 'inform_unknown' and old_state['agent_action'] is not None:
        if old_state['agent_action']['diaact'] == "request":
            for slot in old_state['agent_action']['request_slots']:
                state['request_slots'][slot] = '?'
        elif old_state['agent_action']['diaact'] == "confirm":
            for slot in old_state['agent_action']['inform_slots']:
                del state['inform_slots'][slot] # remove incorrectly recognized slot
                state['request_slots'][slot] = '?'
        # TODO multiple_choice

        return state # should not have informed slot in this case

    # user-requested slots
    req_slot = None
    if sem_frame['intent'].startswith('request'):
        req_slot = sem_frame['intent'][8:]
        if req_slot != 'review':
            state['request_slots'][req_slot] = '?'

    # user-informed slots
    for k, v in sem_frame['slot'].items():
        # FIXME intent might not be always correct...
        '''
        if req_slot is not None and k == req_slot:
            continue
        '''
        if len(v) > 1 or k in ['schedule_str', 'sel_method']:
            # trim suffix for DB query
            if k == 'title' and v.endswith("課"):
                v = v[:-1]
            if k == 'instructor' and (v.endswith("教授") or v.endswith("老師")):
                v = v[:-2]
            if k == 'required_elective' and v.endswith("課"):
                v = v[:-1]

            # classroom slot is seldom recognized correctly so we exclude it
            if k == 'classroom':
                continue

            state['inform_slots'][k] = v

    # move informed slots to constraints
    if old_state['agent_action'] is not None and old_state['agent_action']['diaact'] == "inform":
        for slot in old_state['agent_action']['inform_slots']:
            state['constraints'][slot] = old_state['agent_action']['inform_slots'][slot]
            if slot in state['request_slots']:
                del state['request_slots'][slot]

    return state


@run_once
def multi_turn_lu_setup():
    global understand
    from multiturn_LU.test_MTLU import understand
    print('[Info] Multi-turn LU model loaded.')


# def multi_turn_lu(user_id, sentence):
#     multi_turn_lu_setup()
#     status = understand(user_id, sentence)
#     action = get_action_from_frame(status)
#     return status, action, agent2nl(action)
#     #return status, action, get_NL_from_action(action)


# def multi_turn_lu2(user_id, sentence, reset=False):
#     single_turn_lu_setup()
#     with open('user_log.p', 'rb') as handle:
#         user_log = pickle.load(handle)
#     if reset:
#         user_log[user_id] = {'request_slots': {}, 'inform_slots': {}}
#         with open('user_log.p', 'wb') as handle:
#             pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return
#     status = user_log.get(user_id, {'request_slots': {}, 'inform_slots': {}})
#     d = single_turn_lu(sentence)
#     if 'when' in d['slot']:
#         d['slot']['schedule_str'] = d['slot']['when'][-1]
#         d['slot'].pop('when')
#     if not status['request_slots']:
#         status['request_slots']['schedule_str' if d['intent'] == 'schedule' else d['intent']] = '?'
#     for k, v in d['slot'].items():
#         '''
#         if k not in status['inform_slots']:
#             status['inform_slots'][k] = v
#         '''
#         status['inform_slots'][k] = v # allow updating slots

#     action = get_action_from_frame(status)
#     # return status, action, agent2nl(action)
#     if action['diaact'] in ['inform', 'closing']:
#         user_log[user_id] = {'request_slots': {}, 'inform_slots': {}}
#     else:
#         user_log[user_id] = status
#     with open('user_log.p', 'wb') as handle:
#         pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     return d, status, action, agent2nl(action)
#     #return d, status, action, get_NL_from_action(action)


def set_status(user_id, status=None):
    if not status:  # Generate an id for new status.
        status = {'current_slots': {}, 'request_slots': {}, 'inform_slots': {}, 'constraints': {}, 'group_id': id_generator(), 'user_action': None, 'agent_action': None, 'turn': 0}
    DialogueLogGroup.objects.update_or_create(
        user_id=user_id, group_id=status['group_id'],
        defaults={'status': json.dumps(status, ensure_ascii=False)}
    )
    return status


def get_status(user_id):
    d_groups = DialogueLogGroup.objects.filter(user_id=user_id).order_by('-id')
    # Session live time is 10min
    if not d_groups or (datetime.now() - d_groups[0].time).total_seconds() > 600:
        return set_status(user_id)
    return json.loads(d_groups[0].status)


def multi_turn_lu3(user_id, sentence, reset=False):
    single_turn_lu_setup_new()
    if reset:
        set_status(user_id)
        return
    status = get_status(user_id)
    d = single_turn_lu_new(sentence)
    status = DST_update(status, d)
    # Retrieve reviews
    if d['intent'] == 'request_review':
        review_constraints = {}
        for slot in ['title', 'instructor']:
            if slot in status['inform_slots']:
                review_constraints[slot] = status['inform_slots'][slot] 
            elif slot in status['constraints']:
                review_constraints[slot] = status['constraints'][slot]

        reviews = query_review(review_constraints).order_by('-id')[:20]
        review_resp = []
        if reviews.count() == 0:
            review_resp.append('並未搜尋到相關評價QQ')
        else:
            review_resp.append('幫您搜尋到%d筆相關評價：<br>' % reviews.count())
            for review in reviews:
                review_resp.append('<a target="_blank" href="https://www.ptt.cc/bbs/NTUCourse/%s.html">%s</a><br>' % (review.article_id, review.title))

        set_status(user_id, status)

        return d, status, {}, '\n'.join(review_resp)

    if d['intent'] == 'thanks' or d['intent'] == 'closing':  # Reset dialogue state
        action = {'diaact': 'thanks'}
    elif d['intent'] == 'other' and len(d["slot"]) == 0:
        action = {'diaact': 'inform_unknown'}
    else:
        action = get_action_from_frame(status)

    status['agent_action'] = action

    if action['diaact'] in ['closing', 'thanks']:
        set_status(user_id)
    else:
        set_status(user_id, status)
    return d, status, action, agent2nl(action)


def multi_turn_rl(user_id, sentence, reset=False):
    single_turn_lu_setup_new()
    all_courses = list(query_course({}).values())
    np.random.shuffle(all_courses)
    course_dict = {k: v for k, v in enumerate(all_courses)}
    act_set = text_to_dict('%s/dqn/dia_acts.txt' % settings.BASE_DIR)
    slot_set = text_to_dict('%s/dqn/slot_set.txt' % settings.BASE_DIR)
    agent_params = {}
    agent_params['max_turn'] = 20
    agent_params['epsilon'] = 0.1
    agent_params['agent_run_mode'] = 3
    agent_params['agent_act_level'] = 1
    agent_params['experience_replay_pool_size'] = 200
    agent_params['batch_size'] = 20
    agent_params['gamma'] = 0.9
    agent_params['predict_mode'] = True
    agent_params['trained_model_path'] = '%s/dqn/rl-model.h5' % settings.BASE_DIR
    agent_params['warm_start'] = 2
    agent_params['cmd_input_mode'] = None
    agent_params['model_params'] = {}
    agent = AgentDQN(course_dict, act_set, slot_set, agent_params)

    if reset:
        set_status(user_id)
        return

    status = get_status(user_id)
    semantic_frame = single_turn_lu_new(sentence)

    semantic_frame['diaact'] = semantic_frame['intent']
    semantic_frame['inform_slots'] = semantic_frame['slot']

    status = DST_update(status, semantic_frame)
    status['turn'] += 1  # turn added by user action

    semantic_frame['request_slots'] = status['request_slots']

    status['current_slots'] = {}
    status['current_slots']['inform_slots'] = status['inform_slots']
    status['current_slots']['request_slots'] = status['request_slots']
    status['user_action'] = semantic_frame

    print("lu - Semantic_frame:\n\t", semantic_frame, '\n')
    print("lu - Status:\n\t", status, '\n')

    # Retrieve reviews
    if semantic_frame['intent'] == 'request_review':
        review_constraints = {}
        for slot in ['title', 'instructor']:
            if slot in status['inform_slots']:
                review_constraints[slot] = status['inform_slots'][slot]
            elif slot in status['constraints']:
                review_constraints[slot] = status['constraints'][slot]
        reviews = query_review(review_constraints).order_by('-id')[:20]
        review_resp = []
        if reviews.count() == 0:
            review_resp.append('並未搜尋到相關評價QQ')
        else:
            review_resp.append('幫您搜尋到%d筆相關評價：<br>' % reviews.count())
            for review in reviews:
                review_resp.append(
                    '<a target="_blank" href="https://www.ptt.cc/bbs/NTUCourse/%s.html">%s</a><br>' % (review.article_id, review.title))
        status['agent_action'] = {'diaact': 'inform', 'inform_slots': {}, 'request_slots': {}}
        status['turn'] += 1
        set_status(user_id, status)

        return semantic_frame, status, {}, '\n'.join(review_resp)

    if semantic_frame['intent'] == 'thanks' or semantic_frame['intent'] == 'closing':  # Reset dialogue state
        action = {'diaact': 'thanks'}
    elif semantic_frame['intent'] == 'other' and len(semantic_frame['slot']) == 0:
        action = {'diaact': 'inform_unknown', 'inform_slots': {}, 'request_slots': {}}
    else:
        act_slot_response = agent.feasible_actions[np.argmax(agent.model.predict(
            agent.prepare_state_representation(status)))]
        print("lu - act_slot_response:\n\t", act_slot_response, '\n')
        action = AgentDQN.refine_action(act_slot_response, status)
        print("lu - action:\n\t", action, '\n')

    status['agent_action'] = action

    status['turn'] += 1  # turn added by agent action
    if action['diaact'] in ['closing', 'thanks']:
        set_status(user_id)
    else:
        set_status(user_id, status)
    return semantic_frame, status, action, agent2nl(action)


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
    lu_model = load_model('%s/LU_LSTM/PY3--re_seg.1K+log_extend_1000--bi-LSTM.model' % settings.BASE_DIR)
    print('[Info] Single-turn LU model loaded.')


@run_once
def single_turn_lu_setup_new():  # load new LU models (output new intents)
    global lu_model, idx2label, idx2intent, word2idx

    # load vocab
    obj = json.load(open('%s/LU_LSTM/log1K+template1K.vocab.json' % settings.BASE_DIR, "r"))
    idx2label = obj["slot_vocab"]
    idx2intent = obj["intent_vocab"]
    word2idx = {}
    for i, w in enumerate(obj["word_vocab"]):
        word2idx[w] = i

    # load model
    lu_model = load_model('%s/LU_LSTM/PY3--log1K+template1K--NTUCourse.CWE--LSTM.iw0.8.model' % settings.BASE_DIR)
    print('[Info] Single-turn LU model loaded.')


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
            slot_value_list[slot].append(value)
    for slot in slot_value_list:  # Comparison rule: (1)longer first (2)left first
        max_n_char = 0
        best_value = None
        for value in slot_value_list[slot]:
            n_char = len(list(value))
            if n_char > max_n_char:
                max_n_char = n_char
                best_value = value
        d['slot'][slot] = best_value
    return d
