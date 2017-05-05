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
    courses = []
    for c in all_courses:
        exclude = False
        for slot in dia_state["inform_slots"]:
            if c[slot] != "?" and c[slot] != dia_state["inform_slots"][slot]:
                exclude = True
        if not exclude:
            courses.append(c)
    #############################################
    print ("[INFO] current set of courses")
    print (courses)

    sys_act = {}
    if len(courses) == 1:  # find the unique course
        course = courses[0]
        sys_act["diaact"] = "inform"
        inform_slots = {}
        for slot in dia_state["request_slots"]:
            inform_slots[slot] = course[slot]
        # must provide serial_no to complete the task
        inform_slots["serial_no"] = course["serial_no"]
        sys_act["inform_slots"] = inform_slots
        sys_act["request_slots"] = {}

    elif len(courses) == 0:  # fail
        sys_act["diaact"] = "closing"
        sys_act["inform_slots"] = {}
        sys_act["request_slots"] = {}

    else:
        sys_act["diaact"] = "request"
        req_slot = None
        max_n = 0
        for slot in ["title", "instructor", "classroom", "schedule_str"]:
            # don't ask users something they are asking...
            if slot in dia_state["request_slots"]:
                continue
            # don't ask users something already known
            if slot in dia_state["inform_slots"]:
                continue
            # max # different values --> largest diversity
            n_values = len(set([c[slot] for c in courses]))
            print ("[INFO] slot %s, # values = %d" % (slot, n_values))
            if n_values > max_n:
                max_n = n_values
                req_slot = slot
        sys_act["diaact"] = "request"
        sys_act["inform_slots"] = {}
        sys_act["request_slots"] = {req_slot: "?"}

    return sys_act


def get_NL_from_action(sys_act):
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

    return "".join(res_list)  # TODO


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
    dia_state["request_slots"] = {"instructor": "林智星"}
    dia_state["inform_slots"] = {"classroom": "?"}
    test_dia_states.append(dia_state)

    for dia_state in test_dia_states:
        print ("\n== Dialogue State ==")
        print (dia_state)

        sys_act = get_action_from_frame(dia_state)

        print ("\n== System Action ==")
        print (sys_act)
        NL = get_NL_from_action(sys_act)

        print ("\n== Template-based NLG ==")
        print (NL)

        print ("----------\n")

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

