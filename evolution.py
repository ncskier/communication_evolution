from models import World, Agent, Direction
from view import View
import random
from keras.models import Sequential
from keras.layers import Dense
from deap import tools
import numpy as np
from functools import reduce


class Simulation:
    """Evolution simulation."""

    def __init__(self, draw=True, max_time=25, population_size=50, num_generations=30, world_size=(25,25)):
        self.max_time = max_time
        self.num_generations = num_generations
        # Initialize world
        world_width, world_height = world_size
        self.world = World(world_width, world_height)
        self.num_agents = population_size
        self.world.agents = self.initialize_agents(self.world)
        # Initialize view
        self.draw = draw
        if self.draw:
            self.view = View(self.world)

    def initialize_agents(self, world, nn_weights_list=None):
        """Add agents to [self.world]."""
        # Precondition: [self.world] and [self.num_agents] must be defined
        agents = {}
        for i in range(self.num_agents):
            agent = Agent(i)
            agent.move = bool(random.getrandbits(1))
            agent.direction = Direction(random.randrange(4))
            agent.correct_direction = agent.direction
            agent.model = self.initialize_agent_nn()
            if nn_weights_list:
                agent.model.set_weights(nn_weights_list[i])
            agent.distance = 0
            x = random.randrange(world.width)
            y = random.randrange(world.height)
            while (x, y) in self.world.agents:
                x = random.randrange(world.width)
                y = random.randrange(world.height)
            agents[(x, y)] = agent
        return agents

    def initialize_agent_nn(self):
        """Return neural network for agent."""
        input_vars = 9
        layer_num_neurons = 7
        output_vars = 3
        model = Sequential()
        # Input - Layer
        model.add(Dense(layer_num_neurons, input_shape=(input_vars,), activation='relu', kernel_initializer='uniform'))
        # Hidden - Layers
        model.add(Dense(layer_num_neurons, activation='relu', kernel_initializer='uniform'))
        # model.add(Dense(layer_num_neurons, activation='relu', kernel_initializer='uniform'))
        # Output - Layer
        model.add(Dense(output_vars, activation='sigmoid'))
        return model

    def run(self, retain=0.2, random_select=0.05, mutate=0.01, mutpb=0.6):
        """Run genetic algorithm."""
        for generation in range(0, self.num_generations):
            # Grade - [graded] is list of tuples ([score], model) - higher is better
            self.run_world_simulation(generation)
            print('fitness gen {}:'.format(generation))
            graded = []
            self.fitnesses = []
            for loc in self.world.agents:
                agent = self.world.agents[loc]
                graded.append(
                    (self.fitness(agent, self.world),
                    self.flatten_model_weights(agent.model.get_weights()))
                )
            graded = sorted(graded, key=lambda x: x[0], reverse=True)   # Higher fitness at front
            self.fitnesses = [x[0] for x in graded]
            graded = [x[1] for x in graded]
            # print('{}'.format(graded))
            print('\t{}'.format(self.fitnesses))
            retain_length = int(len(graded)*retain)
            parents = graded[:retain_length]
            # Randomly add other individuals to promote genetic diversity
            for individual in graded[retain_length:]:
                if random_select > random.random():
                    parents.append(individual)
            # Mutate some individuals
            for individual in parents:
                if mutate > random.random():
                    tools.mutation.mutShuffleIndexes(individual[0], mutpb)
            # Crossover parents to create children
            parents_len = len(parents)
            desired_len = len(self.world.agents)
            children = []
            while len(children) < desired_len:
                maleIdx = random.randint(0, parents_len-1)
                femaleIdx = random.randint(0, parents_len-1)
                if maleIdx != femaleIdx:
                    male = (np.copy(parents[maleIdx][0]), parents[maleIdx][1])
                    female = (np.copy(parents[femaleIdx][0]), parents[femaleIdx][1])
                    # print('\tmale_before:', male[0])
                    tools.crossover.cxTwoPoint(male[0], female[0])
                    # print('\tmale_after: ', male[0])
                    children.append(male)
                    if len(children) < desired_len:
                        children.append(female)
            # Create new population
            parents.extend(children)
            nn_weights_list = [self.unflatten_model_weights(x) for x in parents]
            self.world.agents = self.initialize_agents(self.world, nn_weights_list=nn_weights_list)


    def run_world_simulation(self, generation):
        """Run agents through world simulation."""
        draw_path = 'out/test/gen{}/world'.format(generation)
        draw_ext = '.png'
        if self.draw:
            self.view.draw(self.world)
            self.view.save(draw_path+'0'+draw_ext)
        for t in range(1, self.max_time):
            self.update()
            if self.draw:
                self.view.save(draw_path+str(t)+draw_ext)
            # TODO: take this out - this is for a test fitness
            for loc in self.world.agents:
                agent = self.world.agents[loc]
                if agent.moved and agent.direction == agent.correct_direction:
                    agent.distance += 1

    def update(self):
        for loc in self.world.agents:
            self.update_agent(self.world.agents[loc])
        self.world.update()
        if self.draw:
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
        # [7:9] Correct Direction
        correct_direction = [int(i) for i in '{0:02b}'.format(agent.correct_direction.value)]
        return np.concatenate([direction, moved, proximity, correct_direction])

    def parse_nn_output(self, y):
        """Return dictionary decoding numpy array of nn output."""
        output = {}
        # [0:2] Direction
        output['direction'] = Direction(int(''.join(str(i) for i in y[0:2]), 2))
        # [2:3] Move
        output['move'] = bool(y[2])
        return output

    def fitness(self, agent, world):
        """Return fitness of [agent] higher is better than lower."""
        return agent.distance

    def flatten_model_weights(self, weights):
        """Flatten model [weights] with [shapes] into a single numpy array."""
        flat_weights_list = [array.flatten() for array in weights]
        shapes = [array.shape for array in weights]
        return (np.concatenate(flat_weights_list), shapes)

    def unflatten_model_weights(self, individual):
        """Return flattened [weights] into [shapes]."""
        flat_weights, shapes = individual
        weights = []
        pos = 0
        for i in range(0, len(shapes)):
            shape = shapes[i]
            length = reduce(lambda x, y: x*y, shape)
            array = flat_weights[pos:pos+length]
            weights.append(np.reshape(array, shape))
            pos += length
        return weights

    def __str__(self):
        return str(self.world)


def main():
    simulation = Simulation(draw=True)
    simulation.run()

if __name__ == '__main__':
    main()