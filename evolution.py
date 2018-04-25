from models import World, Agent, Direction
from view import View
import random


class Simulation:
    """Evolution simulation."""

    def __init__(self):
        # Initialize world
        self.world = World(25, 25)
        self.num_agents = 50
        self.initialize_agents()
        # Initialize view
        self.view = View(self.world)

    def initialize_agents(self):
        """Add agents to [self.world]."""
        # Precondition: [self.world] and [self.num_agents] must be defined
        for i in range(self.num_agents):
            agent = Agent(i)
            agent.move = bool(random.getrandbits(1))
            agent.direction = Direction(random.randrange(4))
            x = random.randrange(self.world.width)
            y = random.randrange(self.world.height)
            while (x, y) in self.world.agents:
                x = random.randrange(self.world.width)
                y = random.randrange(self.world.height)
            self.world.agents[(x, y)] = agent

    def initialize_agent(self):
        """Return initialized agent for the world."""
        pass

    def update(self):
        self.world.update()
        self.view.draw(self.world)

    def __str__(self):
        return str(self.world)


def main():
    simulation = Simulation()
    simulation.view.draw(simulation.world)
    simulation.view.save('out/world0.png')
    simulation.update()
    simulation.view.save('out/world1.png')

if __name__ == '__main__':
    main()