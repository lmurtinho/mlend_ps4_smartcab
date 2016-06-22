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
        self.reward_sum = 0
        self.disc_reward_sum = 0
        self.n_dest_reached = 0
        self.last_dest_fail = 0
        self.sum_time_left = 0
        self.n_penalties = 0
        self.last_penalty = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # update time and learning rate
        self.time += 1
        learn_rate = 1.0 / self.time
        
        # retrieve best action        
        action = self.next_waypoint

        # On a red light, the agent can only turn right, and even so only if:
        # - no oncoming traffic is going left
        # - no traffic from the left is going forward
        if inputs['light'] == 'red':
            if (action != 'right') or (inputs['oncoming'] == 'left') \
            or (inputs['left'] == 'forward'):
                action = None

        # On a green light, the agent cannot turn left if there is
        # oncoming traffic going forward
        elif inputs['oncoming'] == 'forward' and action == 'left':
            action = None

        # Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'],
                      self.next_waypoint)

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.n_penalties += 1
            print self.state, action
        self.reward_sum += reward
        self.disc_reward_sum += reward / (self.time/10.0)
        
        # Update the q-value of the (state, action) pair
        self.qvals[(self.state, action)] = \
            (1 - learn_rate) * self.qvals.get((self.state, action), 0) + \
            learn_rate * reward
            

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
    results = []
    for i in range(100):
        sim_results = run()
        results.append(run())
    df_results = pd.DataFrame(results)
    df_results.columns = ['reward_sum', 'disc_reward_sum', 'n_dest_reached',
                          'last_dest_fail', 'sum_time_left', 'n_penalties',
                          'last_penalty', 'len_qvals']
    df_results.to_csv('perfect_agent_results.csv')
    print df_results
