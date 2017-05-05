from .kb_helper import KBHelper


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, agent, user):
        self.agent = agent
        self.user = user
        self.kbhelper = KBHelper()
        self.user_action = None
        self.reward = 0
        self.episode_over = False
        self.history_slots = {}
        self.query_slot = 'title'

        # possible answer set
        self.possible_answer = {}

    def initialize_episode(self):
        """ Refresh state for new dialog """
        
        self.reward = 0
        self.episode_over = False
        self.user_action = self.user.initialize_episode()

        for key in self.user_action['inform_slots'].keys():
            self.history_slots[key] = self.user_action['inform_slots'][key]

        self.query_slot = list(self.user.goal['request_slots'].keys())[0]
        self.possible_answer = self.kbhelper.query(self.history_slots, self.query_slot)

        return self.user_action

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        self.user_action, self.episode_over  = self.user.next(self.sys_action)
        
        for key in self.user_action['inform_slots'].keys():
            self.history_slots[key] = self.user_action['inform_slots'][key]

        self.possible_answer = self.kbhelper.query(self.history_slots, self.query_slot)
        self.reward = self.reward_function()

        return (self.episode_over, self.reward)
    

    def reward_function(self):
        """ Reward Function 1: a reward function based on the dialog_status """
        reward = 1
        return reward
    
    
    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
        #print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'],  agent_action['request_slots']))
        print("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'],  user_action['request_slots']))

