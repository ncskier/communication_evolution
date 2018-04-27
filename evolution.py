from models import World, Agent, Direction
from view import View
import random
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class Simulation:
    """Evolution simulation."""

    def __init__(self, max_time=50, population_size=50, num_generations=50, world_size=(25,25)):
        self.max_time = max_time
        self.num_generations = num_generations
        # Initialize world
        world_width, world_height = world_size
        self.world = World(world_width, world_height)
        self.num_agents = population_size
        self.world.agents = self.initialize_agents(self.world)
        # Initialize view
        self.view = View(self.world)

    def initialize_agents(self, world):
        """Add agents to [self.world]."""
        # Precondition: [self.world] and [self.num_agents] must be defined
        agents = {}
        for i in range(self.num_agents):
            agent = Agent(i)
            agent.move = bool(random.getrandbits(1))
            agent.direction = Direction(random.randrange(4))
            agent.model = self.initialize_agent_nn()
            x = random.randrange(world.width)
            y = random.randrange(world.height)
            while (x, y) in self.world.agents:
                x = random.randrange(world.width)
                y = random.randrange(world.height)
            agents[(x, y)] = agent
        return agents

    def initialize_agent_nn(self):
        """Return neural network for agent."""
        input_vars = 7
        layer_num_neurons = 7
        output_vars = 3
        model = Sequential()
        # Input - Layer
        model.add(Dense(layer_num_neurons, input_shape=(input_vars,), activation='relu', kernel_initializer='uniform'))
        # Hidden - Layers
        model.add(Dense(layer_num_neurons, activation='sigmoid', kernel_initializer='uniform'))
        model.add(Dense(layer_num_neurons, activation='sigmoid', kernel_initializer='uniform'))
        # Output - Layer
        model.add(Dense(output_vars, activation='sigmoid'))
        return model

    def run(self, draw=True):
        draw_path = 'out/world'
        draw_ext = '.png'
        self.view.draw(self.world)
        self.view.save(draw_path+'0'+draw_ext)
        for t in range(1, self.max_time):
            self.update(draw=draw)
            if draw:
                self.view.save(draw_path+str(t)+draw_ext)

    def update(self, draw=True):
        for loc in self.world.agents:
            self.update_agent(self.world.agents[loc])
        self.world.update()
        if draw:
            self.view.draw(self.world)

    def update_agent(self, agent):
        """Run agent neural network (model) and update its state."""
        # Create input
        x = self.build_nn_input(agent)
        # Run input on agent model
        predict_y = agent.model.predict(np.array([x]))
        predict_y_int = np.rint(predict_y[0]).astype(int)
        # Parse output & update agent
        output = self.parse_nn_output(predict_y_int)
        agent.direction = output['direction']
        agent.move = output['move']

    def build_nn_input(self, agent):
        """Return input numpy array for agent model."""
        # [0:2] Direction
        direction = [int(i) for i in '{0:02b}'.format(agent.direction.value)]
        # [2:3] Moved
        moved = [int(agent.moved)]
        # [3:7] Proximity Sensors
        proximity = [int(i) for i in agent.proximity]
        return np.concatenate([direction, moved, proximity])

    def parse_nn_output(self, y):
        """Return dictionary decoding numpy array of nn output."""
        output = {}
        # [0:2] Direction
        output['direction'] = Direction(int(''.join(str(i) for i in y[0:2]), 2))
        # [2:3] Move
        output['move'] = bool(y[2])
        return output

    def __str__(self):
        return str(self.world)


def main():
    simulation = Simulation()
    simulation.run()

if __name__ == '__main__':
    main()