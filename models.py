from enum import Enum


class Direction(Enum):
    """Direction in the world."""
    NORTH = 0
    SOUTH = 1
    EAST  = 2
    WEST  = 3


class World:
    """The 1D evolutionary world."""

    def __init__(self, width, height):
        """Initialize state inherent to world/communication"""
        self.width = width
        self.height = height
        self.agents = {}

    def update(self):
        """Update world one step."""
        # (1) Communication between agents
        # (2) Move agents based on their state
        self.update_move()
        # (3) Update agent sensors (maybe do this as it happens?)

    def update_move(self):
        """Move world components one step."""
        # Move agents
        # Move precedence (1) not moving (2) North (3) South (4) East (5) West
        new_agents = {}
        # (1) Not moving
        for loc in self.agents:
            agent = self.agents[loc]
            if (not agent.move):
                new_agents[loc] = agent
        # (2) North
        for loc in self.agents:
            agent = self.agents[loc]
            if (agent.move and agent.direction == Direction.NORTH):
                x, y = loc
                newloc = (x, (y + 1)%self.height)
                if (newloc in self.agents and self.agents[newloc].move and self.agents[newloc].direction == Direction.SOUTH) or (newloc in new_agents):
                    # TODO: update sensor and simplify this code
                    new_agents[loc] = agent
                else:
                    new_agents[newloc] = agent
        # (3) South
        for loc in self.agents:
            agent = self.agents[loc]
            if (agent.move and agent.direction == Direction.SOUTH):
                x, y = loc
                newloc = (x, (y - 1)%self.height)
                if (newloc not in new_agents):
                    new_agents[newloc] = agent
                else:
                    # TODO: update sensor if can't move
                    new_agents[loc] = agent
        # (4) East
        for loc in self.agents:
            agent = self.agents[loc]
            if (agent.move and agent.direction == Direction.EAST):
                x, y = loc
                newloc = ((x + 1)%self.width, y)
                if (newloc in self.agents and self.agents[newloc].move and self.agents[newloc].direction == Direction.WEST) or (newloc in new_agents):
                    # TODO: update sensor and simplify this code
                    new_agents[loc] = agent
                else:
                    new_agents[newloc] = agent
        # (5) West
        for loc in self.agents:
            agent = self.agents[loc]
            if (agent.move and agent.direction == Direction.WEST):
                x, y = loc
                newloc = ((x - 1)%self.width, y)
                if (newloc not in new_agents):
                    new_agents[newloc] = agent
                else:
                    # TODO: update sensor if can't move
                    new_agents[loc] = agent
        # Update to new agents dict
        self.agents = new_agents


    def __str__(self):
        return str(self.agents)

    def __repr__(self):
        return self.__str__()


class Agent:
    """The agents residing in the evolutionary world."""

    def __init__(self):
        """Initialize state inherent to world/communication"""
        # public id (contact id)
        # private id
        self.direction = Direction.NORTH
        self.move = False
        # communication (sequence of bits)
        pass
