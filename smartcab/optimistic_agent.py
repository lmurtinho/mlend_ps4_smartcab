import random
from learning_agent import LearningAgent

class OptimisticAgent(LearningAgent):
    """An optimistic agent that learns to drive in the smartcab world."""

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or one of the best actions, given a state, being
        optimistic in the face of uncertainty.
        """        
        # get all possible q-values for the state
        # (be optimistic in the face of uncertainty)
        all_qvals = {action: self.qvals.get((state, action), 100)
                     for action in self.possible_actions}        
    
        # pick the actions that yield the largest q-value for the state
        best_actions = [action for action in self.possible_actions 
                        if all_qvals[action] == max(all_qvals.values())]
    
        # return one of the best actions at random
        return random.choice(best_actions)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]