import django
import sys
import os
import word2vec
try:
    sys.path.append('../')
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
    django.setup()
    from crawler.models import *
    all_courses = [{k: v for k, v in course.__dict__.items()}
                   for course in Course.objects.filter(semester='105-2')]
    from utils.query import query_course
    from utils.nlg import *
except:
    # some fake courses
    all_courses = [
        {"serial_no": "0001", "title": "機器學習技法", "instructor": "林軒田",
            "classroom": "資102", "schedule_str": "二5,6"},
        {"serial_no": "0002", "title": "機器學習技法", "instructor": "張軒田",
         "classroom": "資102", "schedule_str": "二5,6"},
        {"serial_no": "0003", "title": "機器學習技法", "instructor": "林軒田",
         "classroom": "資103", "schedule_str": "二5,6"},
        {"serial_no": "0004", "title": "機器學習技法", "instructor": "林軒田",
         "classroom": "資102", "schedule_str": "二7,8"},
        {"serial_no": "0005", "title": "機器學習技法", "instructor": "林軒田",
         "classroom": "資204", "schedule_str": "二5,6"}
    ]
    print('Fail to connect to DB, use fake courses instead.')
    print('Please cd to DiaPol-rule folder.')


def get_action_from_frame(dia_state):
    # filter courses according to dia_state
    '''
    courses = []
    for c in all_courses:
        exclude = False
        for slot in dia_state["inform_slots"]:
            if dia_state["inform_slots"][slot] != "?" and c[slot] != dia_state["inform_slots"][slot]:
                exclude = True
        if not exclude:
            courses.append(c)
    '''
    courses = query_course(dia_state["inform_slots"])
    course_ct = courses.count()
    courses = courses.values()

    #############################################
    print ("[INFO] current set of courses: %d" % course_ct)

    # required fields
    sys_act = {} 
    sys_act["diaact"] = "closing"
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
        max_n = 0
        req_max_n = 0 # only consider the slots that can be requested
        #TODO refine a set of slots that the system can request
        for slot in ["title", "instructor", "designated_for", "schedule_str"]:# ordered by priority
            # max # different values --> largest diversity
            values_set = set([c[slot] for c in courses])
            n_values = len(values_set)
            print ("[INFO] slot %s, # values = %d" % (slot, n_values))
            if n_values > max_n: # for checking whether a unique course is found
                max_n = n_values

            if n_values > req_max_n:
                if n_values > 5: # not taking `multiple_choice` action
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

        if max_n <= 1: # only one course satisfy the constraints
            unique_found = True
        elif choice_set is not None and req_max_n <= 5: # no more than 5 values => `multiple_choice`
            sys_act["diaact"] = "multiple_choice"
            sys_act["choice"] = [{req_slot:v} for v in choice_set] # pass list to user
        elif req_slot is not None: # `request`
            sys_act["diaact"] = "request"
            sys_act["request_slots"][req_slot] = "?"

    if unique_found:  # find the unique course
        course = courses[0]
        sys_act["diaact"] = "inform"
        inform_slots = {}
        for slot in dia_state["request_slots"]:
            inform_slots[slot] = course[slot]
        inform_slots["serial_no"] = course["serial_no"] # must provide serial_no to complete the task
        inform_slots["title"] = course["title"] # return course name the ensure the correct course is found
        sys_act["inform_slots"] = inform_slots

    return sys_act


def get_NL_from_action(sys_act): #DEPRECATED!! use utils/nlg.py:agent2nl()
    if sys_act["diaact"] == "closing" and len(sys_act["inform_slots"]) == 0:
        return "不好意思，沒有找到符合條件的課程。"

    res_list = []
    # reponse in a pre-defined order
    for slot in ["serial_no", "title", "instructor", "classroom", "schedule_str"]:
        if slot in sys_act["inform_slots"]:
            if slot == "serial_no":
                res_str = "流水號%s。"
            elif slot == "title":
                res_str = "課名是%s。"
            elif slot == "instructor":
                res_str = "授課教師是%s。"
            elif slot == "classroom":
                res_str = "在%s上課。"
            elif slot == "schedule_str":
                res_str = "%s上課。"
            res_str = res_str % sys_act["inform_slots"][slot]
            res_list.append(res_str)

    # reponse in a pre-defined order
    for slot in ["title", "instructor", "classroom", "schedule_str"]:
        if slot in sys_act["request_slots"]:
            if slot == "title":
                res_str = "請問要找哪門課?"
            elif slot == "instructor":
                res_str = "請問是哪位老師開的?"
            elif slot == "classroom":
                res_str = "請問是在哪上課的?"
            elif slot == "schedule_str":
                res_str = "請問是哪個時間上課的?"
            res_list.append(res_str)

    return "".join(res_list)


if __name__ == '__main__':
    test_dia_states = []
    # 1. 1 request 1 inform
    dia_state = {}
    dia_state["request_slots"] = {"title": "?"}
    dia_state["inform_slots"] = {"schedule_str": "二5,6"}
    test_dia_states.append(dia_state)

    # 2. 1 turn to answer
    dia_state = {}
    dia_state["request_slots"] = {"schedule_str": "?"}
    dia_state["inform_slots"] = {"instructor": "張軒田"}
    test_dia_states.append(dia_state)

    # 3. unsatisfiable
    dia_state = {}
    dia_state["request_slots"] = {"classroom": "?"}
    dia_state["inform_slots"] = {"instructor": "林智星"}
    test_dia_states.append(dia_state)
    
    # 3. multiple choices
    dia_state = {}
    dia_state["request_slots"] = {"schedule_str": "?"}
    dia_state["inform_slots"] = {"title": "機器學習"}
    test_dia_states.append(dia_state)
    
    # 4. confirm
    dia_state = {}
    dia_state["request_slots"] = {"schedule_str": "?"}
    dia_state["inform_slots"] = {"title": "機協", "instructor": "林軒田"}
    test_dia_states.append(dia_state)

    for dia_state in test_dia_states:
        print ("\n== Dialogue State ==")
        print (dia_state)

        sys_act = get_action_from_frame(dia_state)

        print ("\n== System Action ==")
        print (sys_act)

        #NL = get_NL_from_action(sys_act)
        NL = agent2nl(sys_act)
        
        print ("\n== Template-based NLG ==")
        print (NL)
        

        print ("----------\n")

    '''
    ########################
    ### interactive demo ###
    ########################

    print ("\n***** [ interactive demo ] *****")
    import imp
    os.chdir('../multiturn_LU')
    modl = imp.load_source('MTLU', 'model_MTLU.py')
    model_w2v = word2vec.load('word2vec_corpus.bin')
    hist = []#FIXME
    while True:
        sent = input("user>")
        #dia_state = modl.run_MTLU(hist, sent, model_w2v=model_w2v, dim_w2v=modl.dim_w2v)
        dia_state = modl.run_MTLU(hist, sent, model_w2v=model_w2v, dim_w2v=100)
        hist.append(sent)

        print ("\n== Dialogue State ==")
        print (dia_state)

        sys_act = get_action_from_frame(dia_state)

        print ("\n== System Action ==")
        print (sys_act)
        NL = get_NL_from_action(sys_act)

        print ("\n== Template-based NLG ==")
        print (NL)

        print ("----------\n")
    '''
