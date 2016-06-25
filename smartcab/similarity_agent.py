import pandas as pd
import random
from basic_agent import BasicAgent
from environment import Environment
from simulator import Simulator

class LearningAgent(BasicAgent):
    """An agent that learns to drive in the smartcab world."""

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or one of the best actions, given a state.
        """
        # get all possible q-values for the state
        known_states = [key[0] for key in self.qvals.keys()]
        if state in known_states:
            all_qvals = {action: self.qvals.get((state, action), 0)
                         for action in self.possible_actions}
        else:
            similar_state = self.get_similar_state(state)
            all_qvals = {action: self.qvals.get((similar_state, action), 0)
                         for action in self.possible_actions}
        
        # pick the actions that yield the largest q-value for the state
        best_actions = [action for action in self.possible_actions 
                        if all_qvals[action] == max(all_qvals.values())]
        
        # return one of the best actions at random
        return random.choice(best_actions)

    def get_similar_state(self, state):

        similar_state = []
        dif = 1
        
        while (similar_state == []) and dif < len(state):
            similar_states =  [sim_state for sim_state in self.qvals.keys()
                               if sum(map(lambda a, b: a==b, 
                                          state, sim_state)) >= len(state) - dif]
            dif += 1
            
        if not similar_states:
            similar_states = [state]
        
        return random.choice(similar_states)

    def update_qvals(self, state, action, reward):
        """
        Updates the q-value associated with the (state, action) pair
        """
        # define the learning rate for the current time
        learn_rate = 1.0 / self.time
        
        self.qvals[(self.state, action)] = \
            (1 - learn_rate) * self.qvals.get((self.state, action), 0) + \
            learn_rate * reward

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
    results = []
    for i in range(100):
        sim_results = run()
        results.append(sim_results)
    df_results = pd.DataFrame(results)
    df_results.columns = ['n_dest_reached', 'last_dest_fail', 'sum_time_left', 
                          'n_penalties', 'last_penalty', 'len_qvals']
    df_results.to_csv('similarity_agent_results.csv')
    print df_results.describe()