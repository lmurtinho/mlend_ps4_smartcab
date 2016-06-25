import random
from environment import Agent
from planner import RoutePlanner

class BasicAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(BasicAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.qvals = {} # mapping (state, action) to q-values
        self.time = 0 # number of moves performed
        self.possible_actions = (None, 'forward', 'left', 'right')
        self.n_dest_reached = 0 # number of destinations reached
        self.last_dest_fail = 0 # last time agent failed to reach destination
        self.sum_time_left = 0 # sum of time left upon reaching destination over all trials
        self.n_penalties = 0 # number of penalties incurred
        self.last_penalty = 0 # last trial in which the agent incurred in a penalty

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def best_action(self, state):
        """
        Return a random action (other agents will have different policies)
        """
        return random.choice(self.possible_actions)

    def update_qvals(self, state, action, reward):
        """
        Does nothing (in other agents will use reward to update 
        the mapping from (state, action) pairs to q-values)
        """
        self.qvals[(state, action)] = 0
    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # update time
        self.time += 1

        # Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'],
                      self.next_waypoint)

        # Pick an action
        action = self.best_action(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.n_penalties += 1
        
        # Update the q-value of the (state, action) pair
        self.update_qvals(self.state, action, reward)
        
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]