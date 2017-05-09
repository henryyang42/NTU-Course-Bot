import argparse, json, random, copy, re

from django.template import Context, Template

from misc_scripts.generate_template import templates, ask, be


class RuleSimulator():
    """ A rule-based user simulator for testing dialog policy """
    
    def __init__(self, start_set=None):
        """ Constructor shared by all user simulators """
        self.act_set = ['inform', 'request', 'thanks']
        self.slot_set = ['serial_no', 'title', 'instructor', 'classroom', 'schedule_str', 'designated_for', 'required_elective', 'sel_method']
        self.inform_set = copy.deepcopy(self.slot_set)
        self.inform_set.remove('serial_no')
        self.inform_set.append('when')
        self.max_turn = 20
        self.start_set = start_set
        self.request_slot = 'serial_no' # 
        self.reward = 0
        self.accumulated_reward = 0
        self.episodes_num = 0
        self.correct_num = 0
    
    def initialize_episode(self):
        
        self.state = {}
        self.state['history_slots'] = {}
        self.state['history_request_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        
        self.accumulated_reward = self.accumulated_reward + self.reward
        self.episodes_num = self.episodes_num + 1        
        self.reward = 0
        self.episode_over = False
        

        user_action = self._sample_action()

        return user_action  
        
    def _sample_action(self):

        """ randomly sample a start action based on user goal """

        sample_course = random.choice(self.start_set)
        self.ans = {k:sample_course[k] for k in self.slot_set}

        request_type = random.choice([type for type in list(templates.keys()) if 'request' in type])
        # review not support yet :'(
        while request_type == 'request_review':
            request_type = random.choice([type for type in list(templates.keys()) if 'request' in type])
        #################################################

        tpl = random.choice(templates[request_type])

        # Map schedule_str to when
        day = re.findall(r'一|二|三|四|五|六|日', self.ans['schedule_str'])
        time = re.findall(r'\d|[A-D]', self.ans['schedule_str'])
        self.ans['when'] = ('星期'+ day[0] if day else self.ans['schedule_str'])

        # Template NLG
        sample_action = {'nl':tpl.render(Context(self.ans)), 'diaact': request_type, 'inform_slots':{},'request_slots':{}}
        sample_action['request_slots'][request_type.replace('request_','')] = 'UNK'
        self.request_slot = request_type.replace('request_','')

        # Semantic Frame
        for node in tpl.nodelist:
            if node.token.contents in self.inform_set:
                sample_action['inform_slots'][node.token.contents] = self.ans[node.token.contents]
       
        self.state['history_slots'].update(sample_action['inform_slots'])

        # Sample Goal
        self.goal = {'request_slots':sample_action['request_slots']}
        self.goal['inform_slots'] = copy.deepcopy(self.ans)
        self.ans['inform_slots'] = self.ans
        for slot in self.goal['request_slots'].keys():
            del self.goal['inform_slots'][slot]


        return sample_action
    
        
    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.reward = self.reward - 1
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
                self.response_closing(system_action)
            else:
                pass

        if self.state['diaact'] == 'thanks':
            self.reward = self.reward + 100
            self.correct_num = self.correct_num + 1
        elif self.state['diaact'] == 'deny':
            self.reward = self.reward - 100

        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['nl'] = self.user2nl()
        response_action['turn'] = self.state['turn']
        
        return response_action, self.episode_over
    
    
    def response_confirm(self, system_action):
        """ Response for Confirm_Answer (System Action) """
        pass 
            
    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """

        self.episode_over = True
        self.state['diaact'] = 'deny'

    def response_closing(self, system_action):
        """ Response for Thanks (System Action) """

        self.episode_over = True
        self.state['diaact'] = 'deny'

    def response_request(self, system_action):
        """ Response for Request (System Action) """
        
        self.state['diaact'] = "inform"
        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()

        for key in system_action['request_slots'].keys():

            # Penalty.  system request the constraints had been informed before.
            if key in self.state['history_slots'].keys():
                self.reward = self.reward - 20

            # Penalty.  system request the constraints had been informed before.
            if key in self.state['history_request_slots'].keys():
                self.reward = self.reward - 20
                print(self.reward)

            # System request slot is user constraints
            if key in self.goal['inform_slots'].keys():
                self.state['inform_slots'][key] = self.goal['inform_slots'][key]

            # System request slot is user want
            elif key in self.goal['request_slots'].keys():
                self.state['diaact'] = "request"
                self.state["request_slots"][key] = "UNK"

        self.state['history_slots'].update(self.state['inform_slots'])
        self.state['history_request_slots'].update(self.state['request_slots'])



    def response_multiple_choice(self, system_action):
        """ Response for Multiple_Choice (System Action) """
        pass 

    def response_inform(self, system_action):
        """ Response for Inform (System Action) """

        self.state['diaact'] = 'thanks'
        self.episode_over = True

        # System inform slots must match all slots
        for key in system_action['inform_slots'].keys():
            if self.ans['inform_slots'][key] != system_action['inform_slots'][key]:
                self.state['diaact'] = 'deny'

        # System must inform request slot
        for key in self.goal['request_slots'].keys():
            if key in system_action['inform_slots'].keys():
                pass
            else:
                self.state['diaact'] = 'deny'

        # System give serial no. 
        if 'serial_no' in system_action['inform_slots'].keys():
            if self.ans['inform_slots']['serial_no'] == system_action['inform_slots']['serial_no']:
                self.state['diaact'] = 'thanks'
            else:
                self.state['diaact'] = 'deny'

        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()

    def user2nl(self):

        nl_response = 'GG...user2nl壞了'
        random.shuffle(templates['inform'])

        for tpl in templates['inform']:
            
            tpl_keys = [node.token.contents for node in tpl.nodelist if node.token.contents in self.inform_set]
            choose = True
            for slot in self.state['inform_slots'].keys():
                if not slot in tpl_keys:
                    choose = False
                    break
            if choose:
                nl_response = tpl.render(Context(self.ans))
                break

        print(nl_response)
        return nl_response

    def reward_function(self):
        return self.reward
                
    def episodes_reward(self):
        return self.reward, self.accumulated_reward

    def episodes_times(self):
        return self.episodes_num, self.correct_num 


