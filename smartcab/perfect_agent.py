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

        # On a red light, the agent can only turn right, 
        # and only if no left traffic is going forward
        if light == 'red':
            if action != 'right' or left == 'forward':
                action = None

        # On a green light, the agent cannot turn left if there is
        # oncoming traffic going forward or right
        elif action == 'left' and (oncoming == 'forward' or oncoming == 'right'):
            action = None
        
        return action