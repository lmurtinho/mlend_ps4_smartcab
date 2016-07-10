import pandas as pd
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.qvals = {}
        self.time = 0
        self.possible_actions = (None, 'forward', 'left', 'right')
        self.n_penalties = 0
        self.n_dest_reached = 0
        self.sum_time_left = 0
        self.last_dest_fail = 0 # last time agent failed to reach destination
        self.sum_time_left = 0 # sum of time left upon reaching destination over all trials
        self.n_penalties = 0 # number of penalties incurred
        self.last_penalty = 0 # last trial in which the agent incurred in a penalty
        self.states = {} # number of times a state has been visited

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or one of the best actions, given a state.
        """        
        # get all possible q-values for the state
        all_qvals = {action: self.qvals.get((state, action), 0)
                     for action in self.possible_actions}        
        
        # pick the actions that yield the largest q-value for the state
        best_actions = [action for action in self.possible_actions 
                        if all_qvals[action] == max(all_qvals.values())]
        
        # return one of the best actions at random
        return random.choice(best_actions)        

    def update_qvals(self, state, action, reward):
        """
        Updates the q-value associated with the (state, action) pair
        """
        # define the learning rate for the current time
        learn_rate = 1.0 / self.time
        
        self.qvals[(self.state, action)] = \
            (1 - learn_rate) * self.qvals.get((self.state, action), 0) + \
            learn_rate * reward

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
        
        self.states[self.state] = self.states.get(self.state, 0) + 1
        
        # Pick an action
        action = self.best_action(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        
        # print info about penalties
        if reward < 0:
            print "\npenalty!"
            print "light: {0}, oncoming: {1}, left: {2}, waypoint: {3}".format(*self.state)
            print "visit number {} to state".format(self.states[self.state])
            print "action: {}".format(action)
            print "reward: {}\n".format(reward)
        
        # Update the q-value of the (state, action) pair
        self.update_qvals(self.state, action, reward)       
        
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    return sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()