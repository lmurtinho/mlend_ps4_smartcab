from learning_agent import LearningAgent

class NewStateAgent(LearningAgent):
    """An agent that learns to drive in the smartcab world."""

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        # update time and learning rate
        self.time += 1
        
        ok_forward = (inputs['light'] == 'green')
        ok_right = (inputs['light'] == 'green') or \
            (inputs['left'] != 'forward')
        ok_left = all([inputs['light'] == 'green',
                       inputs['oncoming'] != 'forward',
                       inputs['oncoming'] != 'right'])
        
        # Update state
        self.state = (ok_forward, ok_right, ok_left, self.next_waypoint)

        # Pick the best known action
        action = self.best_action(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.n_penalties += 1
        
        # Update the q-value of the (state, action) pair
        self.update_qvals(self.state, action, reward)        
        
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]