from enum import Enum
from sets import Set


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
        next_agents = {}
        moved_set = Set()
        for loc in self.agents:
            rec_set = Set()
            if loc not in moved_set:
                moved_set, next_agents = self.move_agent(loc, rec_set, moved_set, next_agents)
        self.agents = next_agents

    def move_agent(self, loc, rec_set, moved_set, next_agents):
        """Move agent at [loc] to new location and store in [next_agents]."""
        # Precondition: loc in self.agents
        assert (loc in self.agents)
        assert (loc not in moved_set)
        # Return if agent does not want to move
        if not self.agents[loc].move:
            return self.update_agent_loc(loc, loc, moved_set, next_agents)
        # Return if agent is in rec_set
        if loc in rec_set:
            return self.update_agent_loc(loc, loc, moved_set, next_agents)
        # Move critical higher priority agents
        rec_set.add(loc)
        next_loc = self.next_loc(loc)
        crit_locs = self.critical_locs(next_loc)
        for crit_loc in crit_locs:
            if crit_loc == loc:
                continue
            elif (crit_loc in self.agents) and (crit_loc not in moved_set):
                if crit_loc == next_loc:
                    _, next_agents = self.move_agent(crit_loc, rec_set, moved_set, next_agents)
                elif (not self.agents[crit_loc].move) or (self.agents[crit_loc].direction.value < self.agents[loc].direction.value):
                    _, next_agents = self.move_agent(crit_loc, rec_set, moved_set, next_agents)
        # Move agent
        if next_loc in next_agents:
            return self.update_agent_loc(loc, loc, moved_set, next_agents)
        return self.update_agent_loc(loc, next_loc, moved_set, next_agents)

    def update_agent_loc(self, loc, next_loc, moved_set, next_agents):
        """Update agent location from [loc] to [next_loc]."""
        # Precondition: loc in self.agents
        assert (loc in self.agents)
        # Update agent location
        next_agents[next_loc] = self.agents[loc]
        next_agents[next_loc].moved = (loc != next_loc)
        moved_set.add(loc)
        return (moved_set, next_agents)

    def next_loc(self, loc):
        """Calculate next location of agent at [loc]."""
        # Precondition: loc in self.agents
        assert (loc in self.agents)
        # Calculate next location
        agent = self.agents[loc]
        next_loc = loc
        if agent.move:
            x, y = loc
            if agent.direction == Direction.NORTH:
                # North
                next_loc = (x, (y+1) % self.height)
            elif agent.direction == Direction.SOUTH:
                # South
                next_loc = (x, (y-1) % self.height)
            elif agent.direction == Direction.EAST:
                # East
                next_loc = ((x+1) % self.width, y)
            else:
                # West
                next_loc = ((x-1) % self.width, y)
        return next_loc

    def critical_locs(self, loc):
        """Find critical locations where agents could collide with [loc]."""
        x, y = loc
        critical_locs = [loc]
        critical_locs.append(((x+1) % self.width, y))
        critical_locs.append(((x-1) % self.width, y))
        critical_locs.append((x, (y+1) % self.height))
        critical_locs.append((x, (y-1) % self.height))
        return critical_locs

    def __str__(self):
        string = ''
        for loc in sorted(self.agents):
            string += str(loc) + ': ' + str(self.agents[loc]) + '\n'
        return string

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
        self.moved = False
        # communication (sequence of bits)
        pass

    def __str__(self):
        return str(self.direction) + ', move: ' + str(self.move)