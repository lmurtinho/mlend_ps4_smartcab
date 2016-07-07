import pandas as pd
import random
from learning_agent import LearningAgent
from sklearn.tree import DecisionTreeRegressor

class ApproxLearnAgent(LearningAgent):
    """An agent that learns to drive in the smartcab world."""

    def __init__ (self, env):
        super(LearningAgent, self).__init__(env)
        self.df_qv = pd.DataFrame()
        self.rgr = DecisionTreeRegressor()
        self.numbers = {'red': 0, 'green': 1, None: 0, 'left': 1,
                        'right': 2, 'forward': 3}

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or one of the best actions, given a state.
        """
        if self.df_qv.shape[0] < 10:
            return random.choice(self.possible_actions)                         
        
        numeric_state = self.numerify(state)
        
        X = self.df_qv.iloc[:,:-1]
        y = self.df_qv.iloc[:,-1]
        
        # fit classifier
        self.rgr.fit(X, y)
        
        # get all possible sets of variables in state + action
        X_pred = [numeric_state + tuple([self.numbers[action]])
                  for action in self.possible_actions]
        results = self.rgr.predict(X_pred)
        best_numeric_actions = [X_pred[i][-1] for i in range(len(X_pred))
                                if results[i] == max(results)]
        
        best_actions = [action for action in self.possible_actions
                        if self.numbers[action] in best_numeric_actions]        
        
        # return one of the best actions at random
        return random.choice(best_actions)        

    def update_qvals(self, state, action, reward):
        """
        Updates the q-value associated with the (state, action) pair
        """
        # define the learning rate for the current time
        learn_rate = 1.0 / self.time
        numeric_state = self.numerify(state)
        numeric_action = self.numbers[action]
        
        self.qvals[(numeric_state, numeric_action)] = \
            (1 - learn_rate) * self.qvals.get((numeric_state, numeric_action), 
                                              0) + \
            learn_rate * reward
        
        # turn qvals into a dataframe
        self.df_qv = pd.DataFrame([[item for item in pair[0]] + 
                                   [pair[1]] +
                                   [self.qvals[pair]] 
                                   for pair in self.qvals])
                                       
    def numerify(self, state):
        return tuple([self.numbers[item] for item in state])