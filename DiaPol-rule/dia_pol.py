#from crawler.models import *
#from django.db.models import Q

# some fake courses
all_courses = [{"serial_no":"0001", "title":"機器學習技法", "instructor":"林軒田", "classroom":"資102", "schedule_str":"二5,6"}, {"serial_no":"0002", "title":"機器學習技法", "instructor":"張軒田", "classroom":"資102", "schedule_str":"二5,6"}, {"serial_no":"0003", "title":"機器學習技法", "instructor":"林軒田", "classroom":"資103", "schedule_str":"二5,6"}, {"serial_no":"0004", "title":"機器學習技法", "instructor":"林軒田", "classroom":"資102", "schedule_str":"二7,8"}]
#all_courses = Course.objects.filter(semester='105-2')

def get_action_from_frame(dia_state):
    ### filter courses according to dia_state ###
    # TODO should be repalced by Django DB query
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
    if len(courses) == 1: # find the unique course
        course = courses[0]
        sys_act["diaact"] = "inform"
        inform_slots = {}
        for slot in dia_state["request_slots"]:
            inform_slots[slot] = course[slot]
        inform_slots["serial_no"] = course["serial_no"] # must provide serial_no to complete the task
        sys_act["inform_slots"] = inform_slots
        sys_act["request_slots"] = {}

    elif len(courses) == 0: # fail
        sys_act["diaact"] = "closing" 
        sys_act["inform_slots"] = {}
        sys_act["request_slots"] = {}

    else:
        sys_act["diaact"] = "request"
        req_slot = None
        max_n = 0
        for slot in ["title", "instructor", "classroom", "schedule_str"]:
            if slot in dia_state["request_slots"]: # don't ask users something they are asking...
                continue
            if slot in dia_state["inform_slots"]: # don't ask users something already known
                continue
            n_values = len( set( [c[slot] for c in courses] ) ) # max # different values --> largest diversity
            print ("[INFO] slot %s, # values = %d" % (slot, n_values))
            if n_values > max_n:
                max_n = n_values
                req_slot = slot
        sys_act["diaact"] = "request"
        sys_act["inform_slots"] = {}
        sys_act["request_slots"] = {req_slot:"?"}
        
    return sys_act


def get_NL_from_action(sys_act):
    if sys_act["diaact"] == "closing":
        return "不好意思，沒有找到符合條件的課程。"

    res_list = []
    for slot in ["serial_no", "title", "instructor", "classroom", "schedule_str"]: # reponse in a pre-defined order
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

    for slot in ["title", "instructor", "classroom", "schedule_str"]: # reponse in a pre-defined order
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

    return "".join(res_list) #TODO


test_dia_states = []
# 1. 1 request 1 inform
dia_state = {}
dia_state["request_slots"] = {"title":"?"}
dia_state["inform_slots"] = {"schedule_str":"二5,6"}
test_dia_states.append(dia_state)

# 2. 1 turn to answer
dia_state = {}
dia_state["request_slots"] = {"schedule_str":"?"}
dia_state["inform_slots"] = {"instructor":"張軒田"}
test_dia_states.append(dia_state)

# 3. unsatisfiable 
dia_state = {}
dia_state["request_slots"] = {"instructor":"林智星"}
dia_state["inform_slots"] = {"classroom":"?"}
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
