import pandas as pd
import random
from environment import Environment
from simulator import Simulator
from learning_agent import LearningAgent

class LearningRandomAgent(LearningAgent):
    """
    An agent that learns to drive in the smartcab world
    but acts at random sometimes.
    """

    def __init__ (self, env, eps):
        super(LearningRandomAgent, self).__init__(env)
        self.eps = eps

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or a random action
        """
        # get the rate of random values at this point in time
        random_rate = 1.0 - self.time * self.eps
        
        # if random number smaller than random rate, 
        # the agent picks an unexplored action at the current state
        if random.random() < random_rate:
            unexplored_actions = [action for action in self.possible_actions
                                  if (state, action) not in self.qvals.keys()]
            if unexplored_actions:
                actions = unexplored_actions
            else: # if no actions are unexplored in this state, pick any action
                actions = self.possible_actions
        
        else:
            # get all possible q-values for the state
            all_qvals = {action: self.qvals.get((state, action), 0)
                         for action in self.possible_actions}        
        
            # pick the actions that yield the largest q-value for the state
            actions = [action for action in self.possible_actions 
                       if all_qvals[action] == max(all_qvals.values())]
        
        # return one of the actions at random
        return random.choice(actions)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run_random_change(eps):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningRandomAgent, eps)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    return sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

def several_random_changes(epses):
    """
    For each eps value in epses, runs a simulation
    with a LearningRandomAgent.
    Returns a dict with dataframe results for the agent
    for each eps value.
    """
    results = {}
    for eps in epses:
        eps_results = []
        for i in range(100):
            sim_results = run_random_change(eps)
            eps_results.append(sim_results)
        df_results = pd.DataFrame(eps_results)
        df_results.columns = ['n_dest_reached', 'last_dest_fail', 
                              'sum_time_left', 'n_penalties',
                              'last_penalty', 'len_qvals']
        df_results.to_csv("random_rate_{}_results.csv".format(eps))
        results[eps] = df_results
    return results