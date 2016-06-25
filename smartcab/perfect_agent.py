from basic_agent import BasicAgent

class PerfectAgent(BasicAgent):
    """A perfect agent that knows how to drive in the smartcab world."""

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
            if any([action != 'right', oncoming == 'left', 
                    left == 'forward']):
                action = None

        # On a green light, the agent cannot turn left if there is
        # oncoming traffic going forward or right
        elif action == 'left' and (oncoming == 'forward' or oncoming == 'right'):
            action = None
        
        return action            

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]