from utils.query import query_course


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, agent, user):
        self.agent = agent
        self.user = user
        self.user_action = None
        self.reward = 0
        self.episode_times = 0
        self.episode_correct = 0
        self.episode_over = False
        self.query_slot = 'title'

        # possible answer set
        self.possible_answer = {}

    def initialize_episode(self):
        """ Refresh state for new dialog """
        self.history_slots = {}
        self.episode_over = False
        self.episode_times = self.episode_times + 1
        self.user_action = self.user.initialize_episode()

        # User inform slots histroy
        for key in self.user_action['inform_slots'].keys():
            self.history_slots[key] = self.user_action['inform_slots'][key]

        # Retrieve possible answer
        self.query_slot = list(self.user.goal['request_slots'].keys())[0]
        answer_set = list(query_course(self.history_slots).values_list(self.query_slot,flat=True))
        self.possible_answer = {self.query_slot:answer_set,'count':len(answer_set)}

        return self.user_action

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        self.user_action, self.episode_over  = self.user.next(self.sys_action)

        # System Response Correct Answer
        if self.episode_over and self.user_action['diaact']=='thanks':
            self.episode_correct = self.episode_correct + 1
        
        # User inform slots histroy
        for key in self.user_action['inform_slots'].keys():
            self.history_slots[key] = self.user_action['inform_slots'][key]

        # Retrieve possible answer
        answer_set = list(query_course(self.history_slots).values_list(self.query_slot,flat=True))
        self.possible_answer = {self.query_slot:answer_set,'count':len(answer_set)}

        return self.episode_over
    
    
