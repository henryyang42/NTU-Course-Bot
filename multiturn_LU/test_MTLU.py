#encoding=utf-8
import model_MTLU
from model_MTLU import *
import pickle
import sys
sys.path.append("../")
from DiaPol_rule.dia_pol import *
from user_simulator.usersim.usersim_rule import *
from django.db.models import Q
from django.template import Context, Template


request_tpl = {
    'title': [Template('請列出課程名稱'), Template('什麼課')],
    'instructor': [Template('老師的名字'), Template('老師是誰')],
    'schedule_str': [Template('這堂課在星期幾上課?'), Template('上課時間在什麼時候')],
    'classroom': [Template('在哪裡上課'), Template('教室在哪')]
}

inform_tpl = {
    'title': [Template('{{title}}'), Template('課名是{{title}}'), Template('課程名稱為{{title}}')],
    'instructor': [Template('{{instructor}}'), Template('教學老師是{{instructor}}'), Template('教學老師是{{instructor}}'), Template('{{instructor}}上的課'), Template('是{{instructor}}教授')],
    'schedule_str': [Template('{{when}}'), Template('上課時間是{{when}}'), Template('我想上{{when}}的課')],
    'classroom': [Template('{{classroom}}'), Template('上課教室是{{classroom}}'), Template('在{{classroom}}上課'), Template('在{{classroom}}')]
}


def sem2nl(sem_in):
    if sem_in['diaact'] == 'request':
        attr = next(iter(sem_in['request_slots']))
        tpl = random.choice(request_tpl[attr])
        return tpl.render(Context(sem_in['request_slots']))
    elif sem_in['diaact'] == 'inform':
        attr = next(iter(sem_in['inform_slots']))
        tpl = random.choice(inform_tpl[attr])
        return tpl.render(Context(sem_in['inform_slots']))
    else:
        return '謝謝！'


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
    all_courses = Course.objects.filter(~Q(classroom=''),~Q(instructor=''), semester='105-2').all().values()
    '''uid1 = 123
    s1 = '課名是自然語言處理'
    s2 = '教室在哪'
    s_list1 = [s1, s2]
    user_sim = RuleSimulator(all_courses)
    user_sim.initialize_episode()
    for s in s_list1:
        status = understand(uid1, s)
        action = get_action_from_frame(status)
        sem_frame, over = user_sim.next(action)
        print('Status:', status)
        print('Action:', action)
        print('sem_fram:', sem_frame, over)
    print ('=====================')
    uid1 = 456
    s1 = '課程名稱是道教文化專題研究'
    s2 = '老師是誰?'
    s_list1 = [s1, s2]
    user_sim = RuleSimulator(all_courses)
    user_sim.initialize_episode()
    for s in s_list1:
        status = understand(uid1, s)
        action = get_action_from_frame(status)
        sem_frame, over = user_sim.next(action)
        print('Status:', status)
        print('Action:', action)
        print('sem_fram:', sem_frame, over)
'''
    uid1 = 123
    user_sim = RuleSimulator(all_courses)
    user_action = user_sim.initialize_episode()
    print('-----# Turn 1-----')
    print('User_action:', user_action)
    
    user_sentence = sem2nl(user_action)
    status = understand(uid1, user_sentence)
    system_action = get_action_from_frame(status)
    print('-----# Turn 2-----')
    print('Status:', status)
    print('System Action:', system_action)
    
    user_action = user_sim.next(system_action)[0]
    user_sentence = sem2nl(user_action)
    status = understand(uid1, user_sentence)
    system_action = get_action_from_frame(status)
    print('-----# Turn 3-----')
    print('Status:', status)
    print('User Action:', user_action)
    print('System Action:', system_action)
   
    user_action = user_sim.next(system_action)[0]
    user_sentence = sem2nl(user_action)
    status = understand(uid1, user_sentence)
    system_action = get_action_from_frame(status)
    print('-----# Turn 4-----')
    print('Status:', status)
    print('User Action:', user_action)
    print('System Action:', system_action)

    user_action = user_sim.next(system_action)[0]
    user_sentence = sem2nl(user_action)
    status = understand(uid1, user_sentence)
    system_action = get_action_from_frame(status)
    print('-----# Turn 5-----')
    print('Status:', status)
    print('User Action:', user_action)
    print('System Action:', system_action)

