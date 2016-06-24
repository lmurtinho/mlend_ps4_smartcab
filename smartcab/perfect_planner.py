import random

class RoutePlanner(object):
    """Silly route planner that is meant for a perpendicular grid network."""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.destination = None

    def route_to(self, destination=None):
        self.destination = destination if destination is not None else random.choice(self.env.intersections.keys())
        #print "RoutePlanner.route_to(): destination = {}".format(destination)  # [debug]

    def get_delta(self):
        """
        get the horizontal and vertical distance
        between location and destination.
        """
        location = self.env.agent_states[self.agent]['location']
        grid_size = self.env.grid_size
        heading = self.env.agent_states[self.agent]['heading']
        delta = [0, 0]
        for i in range(2):
            # 1st option: destination to the east/south of location
            if self.destination[i] > location[i]:
                # two possible distances, going east/south or
                # going west/north
                possible_delta = [self.destination[i] - location[i], 
                                  location[i] + grid_size[i] - self.destination[i]]
                # if both are the same, check the heading
                if possible_delta[0] == possible_delta[1]:
                    # if agent is turned toward this axis,
                    # pick direction it is turned to
                    if heading[i]:
                        delta[i] = possible_delta[0] * heading[i]
                    # if it isn't, pick the distance to the right
                    # (it's easier to turn right than left)
                    elif i: # north/south axis: negate heading to go right
                        delta[i] = possible_delta[0] * -sum(heading)
                    else: # west/east axis: right is same sign as heading
                        delta[i] = possible_delta[0] * sum(heading)
                # if the first distance is the smallest, pick it
                elif min(possible_delta) == possible_delta[0]:
                    delta[i] = possible_delta[0]
                # if seconde distance is the smallest, pick minus it
                # (the agent will have to go west/north to get to a point
                #  to the east/south)
                else:
                    delta[i] = -possible_delta[1]
            # 2nd option: destination to the west/north of location
            else:
                # two possible distances, going west/north or
                # going east/south
                possible_delta = [location[i]-self.destination[i], self.destination[i] + grid_size[i] - location[i]]
                # if both are the same, check the heading                
                if possible_delta[0] == possible_delta[1]:
                    # if agent is turned toward this axis,
                    # pick direction it is turned to
                    if heading[i]:
                        delta[i] = abs(possible_delta[0]) * heading[i]
                    # if it isn't, pick the distance to the right
                    # (it's easier to turn right than left)
                    elif i: # north/south axis: negate heading to go right
                        delta[i] = possible_delta[0] * -sum(heading)
                    else: # west/east axis: right is same sign as heading
                        delta[i] = possible_delta[0] * sum(heading)
                # if the first distance is the smallest, pick minus it
                # (the agent will have to go west/north)
                elif min(possible_delta) == possible_delta[0]:
                    delta[i] = -possible_delta[0]
                # if second distance is smallest, pick it
                else:
                    delta[i] = possible_delta[1]
        # return a tuple
        return tuple(delta)

    def next_waypoint(self):
        """
        Calculate the next waypoint.
        """
        # get delta from destination
        self.delta = self.get_delta()
        delta = self.delta

        # get agent heading        
        heading = self.env.agent_states[self.agent]['heading']
        
        # if agent is turned to the east/west axis
        if heading[0]:
            # if it needs to go forward, do it
            if delta[0] * heading[0] > 0:
                return 'forward'
            # else check if it needs to go backward
            elif delta[0] * heading[0] < 0:
                if delta[1] * heading[0] > 0:
                    return 'right'
                else:
                    return 'left'
            elif delta[1] * heading[0] > 0:
                return 'right'
            else:
                return 'left'
        else:
            if delta[1] * heading[1] > 0:
                return 'forward'
            elif delta[1] * heading[1] < 0:
                if delta[0] * heading[1] < 0:
                    return 'right'
                else:
                    return 'left'
            elif delta[0] * heading[1] < 0:
                return 'right'
            else:
                return 'left'
