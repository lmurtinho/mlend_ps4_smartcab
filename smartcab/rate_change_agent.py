import pandas as pd
from environment import Environment
from simulator import Simulator
from learning_agent import LearningAgent

class RateChangeAgent(LearningAgent):
    """
    An agent that learns to drive in the smartcab world
    and whose learning rate can be tweaked.
    """
    
    def __init__ (self, env, mult):
        super(RateChangeAgent, self).__init__(env)
        self.mult = mult
    
    def update_qvals(self, state, action, reward):
        """
        Updates the q-value associated with the (state, action) pair
        """
        # define the learning rate for the current time
        learn_rate = 1.0 / (1 + self.mult*self.time)
        
        self.qvals[(self.state, action)] = \
            (1 - learn_rate) * self.qvals.get((self.state, action), 0) + \
            learn_rate * reward

def run_rate_change(mult):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(RateChangeAgent, mult)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    return sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

def several_rate_changes(mults, folder):
    """
    For each mult value in mults, runs a simulation
    with a RateChangeAgent.
    Returns a dict with dataframe results for the agent
    for each mult value.
    """
    results = {}
    for mult in mults:
        mult_results = []
        for i in range(100):
            sim_results = run_rate_change(mult)
            mult_results.append(sim_results)
        df_results = pd.DataFrame(mult_results)
        df_results.columns = ['n_dest_reached', 'last_dest_fail', 
                              'sum_time_left', 'n_penalties',
                              'last_penalty', 'len_qvals']
        df_results.to_csv("{}/rate_change_{}_results.csv".format(folder, mult))
        results[mult] = df_results
    return results