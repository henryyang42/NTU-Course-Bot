#encoding=utf-8
import model_MTLU
from model_MTLU import *
import pickle

with open('user_log.pickle', 'wb') as handle:
    pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def understand(user_id, sentence):
    print('understand', sentence)
    with open('user_log.pickle', 'rb') as handle:
        user_log = pickle.load(handle)

    user = user_log.get(user_id, {'history': np.zeros((1,len_history,len_sentence,dim_w2v)), 'state': {'request_slots': {}, 'inform_slots': {}} })
    # Load
    history = user['history']
    old_state = user['state']
    print('User log:', old_state)
    # Save
    sentence_str = ' '.join(list(jieba.cut(sentence)))
    sentence_vec = sentence_to_vec(sentence_str, model_w2v, dim_w2v=dim_w2v)
    sentence_vec = sentence_vec.reshape((1,len_sentence,dim_w2v))
    user['history'] = history_add(history,sentence_vec,len_history)
    user['state'] = run_MTLU(history=history, sentence=sentence, old_state=old_state,
                 model_w2v=model_w2v, len_history=len_history, len_sentence=len_sentence,
                 dim_w2v=dim_w2v, dim_after_rnn=dim_after_rnn, num_tag=num_tag, dim_status=dim_status)
    user_log[user_id] = user
    with open('user_log.pickle', 'wb') as handle:
        pickle.dump(user_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return user['state']

if __name__ == '__main__':
    uid = 123
    s1 = '課名是MATLAB及其應用'
    s2 = '這堂課在星期幾上課 ?'
    s3 = '我想上星期三的課'
    s4 = '丁亮上的課'
    s5 = '上課時間是星期五'
    s6 = '什麼課'

    # status1 = run_MTLU(history=None, sentence=s1, old_state=None,
    #              model_w2v=None, len_history=len_history, len_sentence=len_sentence,
    #              dim_w2v=dim_w2v, dim_after_rnn=dim_after_rnn, num_tag=num_tag, dim_status=dim_status)
    status1 = understand(uid, s1)
    print(status1)

    # status2 = run_MTLU(history=None, sentence=s2, old_state=status1,
    #              model_w2v=None, len_history=len_history, len_sentence=len_sentence,
    #              dim_w2v=dim_w2v, dim_after_rnn=dim_after_rnn, num_tag=num_tag, dim_status=dim_status)
    status2 = understand(uid, s2)
    print(status2)
    status3 = understand(uid, s3)
    print(status3)
    status4 = understand(uid, s4)
    print(status4)
    status5 = understand(uid, s5)
    print(status5)
    status6 = understand(uid, s6)
    print(status6)
