from .usersim import UserSimulator
import argparse, json, random, copy



class RuleSimulator():
    """ A rule-based user simulator for testing dialog policy """
    
    def __init__(self, start_set=None):
        """ Constructor shared by all user simulators """
        self.act_set = ['inform', 'request', 'thanks']
        self.slot_set = ['serial_no', 'title', 'instructor', 'classroom', 'schedule_str']
        self.max_turn = 50
        self.start_set = start_set

    
    def initialize_episode(self):
        
        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        
        self.episode_over = False
        self.goal = self._sample_goal(self.start_set)
        self.ans = copy.deepcopy(self.goal)

        # Our Task
        # first element is serial_no
        request_slot = random.choice(self.slot_set[1:])
        self.goal['request_slots'][request_slot] = 'UNK'
        del self.goal['inform_slots'][request_slot]

        user_action = self._sample_action()

        return user_action  
        
    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        action = random.choice(['inform', 'request'])
        sample_action = {'diaact':action, 'inform_slots':{}, 'request_slots':{}}
        
        if action == 'inform':
            sample_slots = list(self.goal['inform_slots'].keys())
            if 'serial_no' in sample_slots:
                sample_slots.remove('serial_no')
            slot = random.choice(sample_slots)
            sample_action['inform_slots'][slot] = self.goal['inform_slots'][slot]
        elif action == 'request':
            sample_action['request_slots'] = self.goal['request_slots']
        else:
            pass

        sample_action['turn'] = self.state['turn']
        
        return sample_action

    
    def _sample_goal(self, goal_set):
        """ sample a user goal  """
        
        sample_course = random.choice(self.start_set)
        sample_goal = {'inform_slots':{key:sample_course[key] for key in self.slot_set}, 'request_slots':{}}
        print(sample_goal)

        return sample_goal

        
    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.episode_over = False
        sys_act = system_action['diaact']
        
        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            if sys_act == "inform":
                self.response_inform(system_action)
            elif sys_act == "request":
                self.response_request(system_action) 
            elif sys_act == "thanks":
                self.response_thanks(system_action)
            elif sys_act == "closing":
                self.episode_over = True
                self.state['diaact'] = "GG"
            else:
                pass
        
        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        
        return response_action, self.episode_over
    

    def response_confirm_answer(self, system_action):
        """ Response for Confirm_Answer (System Action) """
        pass 

            
    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """

        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()
        self.episode_over = True


    def response_request(self, system_action):
        """ Response for Request (System Action) """
        
        self.state['diaact'] = "inform"
        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()

        for key in system_action['request_slots'].keys():
            if key in self.goal['inform_slots'].keys():
                self.state['inform_slots'][key] = self.goal['inform_slots'][key]
            elif key in self.goal['request_slots'].keys():
                self.state['diaact'] = "request"
                self.state["request_slots"][key] = "UNK"


    def response_multiple_choice(self, system_action):
        """ Response for Multiple_Choice (System Action) """
        pass 


    def response_inform(self, system_action):
        """ Response for Inform (System Action) """

        self.state['diaact'] = 'thanks'
        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()
        self.episode_over = True

        for key in system_action['inform_slots'].keys():
            if self.ans['inform_slots'][key] != system_action['inform_slots'][key]:
                self.state['diaact'] = 'deny'
                

