import pandas as pd
from environment import Environment
from simulator import Simulator
from basic_agent import BasicAgent

class PerfectAgent(BasicAgent):
    """An agent that learns to drive in the smartcab world."""

    def best_action(self, state):
        """
        Returns the best possible action.
        """        
        # retrieve state information
        light, oncoming, left, waypoint = state

        # retrieve best action        
        action = waypoint

        # On a red light, the agent can only turn right, and even so only if:
        # - no oncoming traffic is going left
        # - no traffic from the left is going forward
        if light == 'red':
            if (action != 'right') or (oncoming == 'left') \
            or (oncoming == 'forward') or (oncoming == 'forward'):
                action = None

        # On a green light, the agent cannot turn left if there is
        # oncoming traffic going forward or right
        elif action == 'left' and (oncoming == 'forward' or oncoming == 'right'):
            action = None
        
        return action            

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(PerfectAgent)  # create agent
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
        results.append(run())
    df_results = pd.DataFrame(results)
    df_results.columns = ['reward_sum', 'disc_reward_sum', 'n_dest_reached',
                          'last_dest_fail', 'sum_time_left', 'n_penalties',
                          'last_penalty', 'len_qvals']
    df_results.to_csv('perfect_agent_results.csv')
    # print df_results
