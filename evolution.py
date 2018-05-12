from models import World, Agent, Direction
from view import View
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from deap import tools
import numpy as np
from functools import reduce
import sys
import os


class Simulation:
    """Evolution simulation."""

    def __init__(self, path='out/test/', draw=False, max_time=50, population_size=35, num_generations=20, world_size=(25,25)):
        self.max_time = max_time
        self.num_generations = num_generations
        self.generation = 0
        self.recurrent_nodes = 10
        # Create path
        self.path = path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        stats_path = '{}stats.csv'.format(self.path)
        with open(stats_path, 'w') as f:
            f.write('population size, {}\n'.format(population_size))
            f.write('world simulation time, {}\n'.format(max_time))
            f.write('world size, {} by {}\n'.format(world_size[0], world_size[1]))
            f.write('\ngen, mean, max, min, fitness one agent worlds, fitness description, nn description')
        # Initialize world
        world_width, world_height = world_size
        self.world = World(world_width, world_height)
        self.num_agents = population_size
        self.initialize_agents(self.world)
        # Initialize view
        self.draw = draw
        if self.draw:
            self.view = View(self.world)

    def initialize_teams(self):
        """Initialize list of teams."""
        teams = []
        for i in range(self.num_agents):
            if i < self.num_agents/2:
                teams.append(0)
            else:
                teams.append(1)
        random.shuffle(teams)
        return teams

    def initialize_agents(self, world, nn_weights_list=None):
        """Add agents to [self.world]."""
        # Precondition: [self.world] and [self.num_agents] must be defined
        world.agents = {}
        teams = self.initialize_teams()
        for i in range(self.num_agents):
            nn_weights = None
            if nn_weights_list:
                nn_weights = nn_weights_list[i]
            self.initialize_agent(world, i, teams[i], nn_weights=nn_weights)

    def initialize_agent(self, world, i, team, nn_model=None, nn_weights=None):
        agent = Agent(i)
        agent.move = bool(random.getrandbits(1))
        agent.direction = Direction(random.randrange(4))
        agent.team = team
        agent.recurrent_memory = np.array(self.recurrent_nodes*[0.0])
        if nn_model:
            agent.model = nn_model
        else:
            agent.model = self.initialize_agent_nn()
            if nn_weights:
                agent.model.set_weights(nn_weights)
        x = random.randrange(world.width)
        y = random.randrange(world.height)
        while (x, y) in world.agents:
            x = random.randrange(world.width)
            y = random.randrange(world.height)
        world.agents[(x, y)] = agent
        return (x, y)

    def run(self, world, one=False, retain=0.2, random_select=0.05, mutate=0.01, mutpb=0.6):
        """Run genetic algorithm on [world]."""
        for generation in range(self.generation, self.num_generations):
            # Grade - [graded] is list of tuples ([score], model) - higher is better
            if one:
                self.run_world_simulation_one(generation)
            else:
                self.run_world_simulation(world, generation)
            print('fitness gen {}:'.format(generation))
            graded = self.evaluate_fitnesses(world, one)
            # print('{}'.format(graded))
            print('\t{}'.format(self.fitnesses))
            retain_length = int(len(graded)*retain)
            parents = graded[:retain_length]
            # Save generation with fitness
            self.save_generation(generation, one)
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
            desired_len = len(world.agents)
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
            self.initialize_agents(world, nn_weights_list=nn_weights_list)

    def run_world_simulation(self, world, generation, one_loc=None):
        """Run agents through world simulation."""
        draw_path = '{}gen{}/world'.format(self.path, generation)
        if one_loc:
            one_agent = self.world.agents[one_loc]
            draw_path = '{}gen{}/agent-i{}/world'.format(self.path, generation, one_agent.private_id)
        draw_ext = '.png'
        if self.draw:
            self.view.draw(world)
            self.view.save(draw_path+'0'+draw_ext)
        # print('max_time: {}'.format(self.max_time))
        for t in range(1, self.max_time):
            self.update(world)
            if self.draw:
                self.view.save(draw_path+str(t)+draw_ext)

    def run_world_simulation_one(self, generation):
        """Run agents through world simulations of only themselves."""
        for loc in self.world.agents:
            agent = self.world.agents[loc]
            model = agent.model
            # Compose world of agent at loc
            world = World(self.world.width, self.world.height)
            world.agents = {}
            teams = self.initialize_teams()
            for i in range(self.num_agents):
                self.initialize_agent(world, i, teams[i], nn_model=model)
            # Run simulation
            self.run_world_simulation(world, generation, one_loc=loc)
            # Calculate and save fitnesses in [agent]
            agent.fitness = 0.0
            for loc in world.agents:
                agent.fitness += self.fitness(loc, world)

    def evaluate_fitnesses(self, world, one=False):
        """Evaluate fitnesses to [self.fitnesses]."""
        graded = []
        self.fitnesses = []
        for loc in world.agents:
            agent = world.agents[loc]
            graded.append(
                (loc,
                self.fitness(loc, world, one=one),
                self.flatten_model_weights(agent.model.get_weights()))
            )
        graded = sorted(graded, key=lambda x: x[1], reverse=True)   # Higher fitness at front
        self.fitnesses = [x[1] for x in graded]
        self.private_ids = [world.agents[x[0]].private_id for x in graded]
        graded = [x[2] for x in graded]
        return graded

    def visualize(self, world, one=False):
        """Save visualization of generation."""
        draw = self.draw
        self.draw = True
        self.view = View(world)
        if one:
            self.run_world_simulation_one(self.generation)
        else:
            self.run_world_simulation(world, self.generation)
            self.evaluate_fitnesses(world)
            # Save fitness
            print('fitness gen {}:'.format(self.generation))
            print('\t{}'.format(self.fitnesses))
        self.draw = draw

    def update(self, world):
        for loc in world.agents:
            self.update_agent(world.agents[loc])
        world.update()
        if self.draw:
            self.view.draw(world)

    # VARIABLE
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
        agent.recurrent_memory = output['recurrent_memory']

    # # VARIABLE -------------------------------------------------------------
    # # Recurrent nodes & proximity team
    # def initialize_agent_nn(self):
    #     """Return neural network for agent."""
    #     self.nn_name = 'know team and has recurrent nodes'
    #     input_vars = 12 + self.recurrent_nodes
    #     layer_num_neurons = 12 + self.recurrent_nodes
    #     output_vars = 3 + self.recurrent_nodes
    #     model = Sequential()
    #     # Input - Layer
    #     model.add(Dense(layer_num_neurons, input_shape=(input_vars,), activation='relu', kernel_initializer='uniform'))
    #     # Hidden - Layers
    #     model.add(Dense(layer_num_neurons, activation='relu', kernel_initializer='uniform'))
    #     # model.add(Dense(layer_num_neurons, activation='relu', kernel_initializer='uniform'))
    #     # Output - Layer
    #     model.add(Dense(output_vars, activation='sigmoid'))
    #     return model

    # # VARIABLE
    # def build_nn_input(self, agent):
    #     """Return input numpy array for agent model."""
    #     # [0:2] Direction
    #     direction = [int(i) for i in '{0:02b}'.format(agent.direction.value)]
    #     # [2:3] Moved
    #     moved = [int(agent.moved)]
    #     # [3:7] Proximity Sensors
    #     proximity = [int(i) for i in agent.proximity]
    #     # [7:11] Team Proximity Sensors
    #     team_proximity = [int(i) for i in agent.team_proximity]
    #     # [11:12] Team
    #     team_id = [int(i) for i in '{0:1b}'.format(agent.team)]
    #     # [12:12+self.recurrent_nodes]
    #     recurrent = agent.recurrent_memory
    #     return np.concatenate([direction, moved, proximity, team_proximity, team_id, recurrent])

    # # VARIABLE
    # def parse_nn_output(self, y):
    #     """Return dictionary decoding numpy array of nn output."""
    #     output = {}
    #     # [0:2] Direction
    #     output['direction'] = Direction(int(''.join(str(i) for i in y[0:2]), 2))
    #     # [2:3] Move
    #     output['move'] = bool(y[2])
    #     # [3:3+self.recurrent_nodes]
    #     output['recurrent_memory'] = y[3:3+self.recurrent_nodes]
    #     return output

    # VARIABLE -------------------------------------------------------------
    # Recurrent nodes & No knowledge of other teams
    def initialize_agent_nn(self):
        """Return neural network for agent."""
        self.nn_name = 'NO knowledge of others team and has recurrent nodes. Knows own team'
        input_vars = 8 + self.recurrent_nodes
        layer_num_neurons = 8 + self.recurrent_nodes
        output_vars = 3 + self.recurrent_nodes
        model = Sequential()
        # Input - Layer
        model.add(Dense(layer_num_neurons, input_shape=(input_vars,), activation='relu', kernel_initializer='uniform'))
        # Hidden - Layers
        model.add(Dense(layer_num_neurons, activation='relu', kernel_initializer='uniform'))
        # model.add(Dense(layer_num_neurons, activation='relu', kernel_initializer='uniform'))
        # Output - Layer
        model.add(Dense(output_vars, activation='sigmoid'))
        return model

    # VARIABLE
    def build_nn_input(self, agent):
        """Return input numpy array for agent model."""
        # [0:2] Direction
        direction = [int(i) for i in '{0:02b}'.format(agent.direction.value)]
        # [2:3] Moved
        moved = [int(agent.moved)]
        # [3:7] Proximity Sensors
        proximity = [int(i) for i in agent.proximity]
        # [7:8] Team
        team_id = [int(i) for i in '{0:1b}'.format(agent.team)]
        # [8:8+self.recurrent_nodes]
        recurrent = agent.recurrent_memory
        return np.concatenate([direction, moved, proximity, team_id, recurrent])

    # VARIABLE
    def parse_nn_output(self, y):
        """Return dictionary decoding numpy array of nn output."""
        output = {}
        # [0:2] Direction
        output['direction'] = Direction(int(''.join(str(i) for i in y[0:2]), 2))
        # [2:3] Move
        output['move'] = bool(y[2])
        # [3:3+self.recurrent_nodes]
        output['recurrent_memory'] = y[3:3+self.recurrent_nodes]
        return output

    # VARIABLE -------------------------------------------------------------
    # def fitness(self, loc, world, one=False):
    #     """Return fitness of agent at [loc] higher is better than lower."""
    #     self.fitness_name = 'imperfect match fitness - points for facing agent not facing this agent'
    #     if one:
    #         return world.agents[loc].fitness / len(world.agents)
    #     fitness = 0
    #     agent = world.agents[loc]
    #     next_loc = world.next_loc(loc, agent.direction)
    #     if next_loc in world.agents:
    #         fitness += 5
    #         adjacent_agent = world.agents[next_loc]
    #         if agent.team != adjacent_agent.team:
    #             fitness += 10
    #             if loc == world.next_loc(next_loc, adjacent_agent.direction):
    #                 fitness += 5
    #     world.agents[loc].fitness = fitness
    #     return fitness

    # VARIABLE -------------------------------------------------------------
    # only (+) fitness if facing another agent from other team who is facing this agent
    def fitness(self, loc, world, one=False):
        """Return fitness of agent at [loc] higher is better than lower."""
        self.fitness_name = 'only perfect match fitness - this agent facing an opposite team agent facing this agent'
        if one:
            return world.agents[loc].fitness / len(world.agents)
        fitness = 0
        agent = world.agents[loc]
        next_loc = world.next_loc(loc, agent.direction)
        if next_loc in world.agents:
            adjacent_agent = world.agents[next_loc]
            if agent.team != adjacent_agent.team:
                if loc == world.next_loc(next_loc, adjacent_agent.direction):
                    fitness += 20
        world.agents[loc].fitness = fitness
        return fitness

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

    def save_generation(self, generation, one):
        """Save generation data to [self.path]/genXX/"""
        # Set the path
        generation_path = '{}gen{}/'.format(self.path, generation)
        # Create directory if it does not exist
        if not os.path.exists(os.path.dirname(generation_path)):
            os.makedirs(os.path.dirname(generation_path))
        # Save model to .json
        model_path = '{}model.json'.format(generation_path)
        with open(model_path, 'w') as f:
            f.write(self.world.agents[list(self.world.agents.keys())[0]].model.to_json())
        # Save all agent genomes (weights)
        for loc in self.world.agents:
            agent = self.world.agents[loc]
            agent_path = '{}agent-i{}.hdf5'.format(generation_path, agent.private_id)
            self.world.agents[loc].model.save_weights(agent_path)
        # Save fitness
        f_mean = np.mean(self.fitnesses)
        f_max = np.amax(self.fitnesses)
        f_min = np.amin(self.fitnesses)
        fitness_path = '{}fitness.csv'.format(generation_path)
        with open(fitness_path, 'w') as f:
            f.write('private_id,{}'.format(','.join(map(str, self.private_ids))))
            f.write('\nfitnesses,{}'.format(','.join(map(str, self.fitnesses))))
            f.write('\nmean,{}'.format(f_mean))
        # Save stats
        stats_path = '{}stats.csv'.format(self.path)
        with open(stats_path, 'a') as f:
            f.write('\n{}, {:.3f}, {:.3f}, {:.3f}, {}, {}, {}'.format(generation, f_mean, f_max, f_min, one, self.fitness_name, self.nn_name))

    def load_generation(self, generation=-1):
        """Load generation data from [self.path]/gen[generation]/"""
        # Set path - if no generation specified, load the last one
        if (generation == -1):
            gen_dir = sorted(os.listdir(self.path))[-1]
            generation_path = '{}{}/'.format(self.path, gen_dir)
            self.generation = int(gen_dir[3:])
        else:
            generation_path = '{}gen{}/'.format(self.path, generation)
            self.generation = generation
        # Iterate over agents
        self.world.agents.clear()
        teams = self.initialize_teams()
        model_path = '{}model.json'.format(generation_path)
        for name in os.listdir(generation_path):
            if not ('agent' in name and 'hdf5' in name):
                continue
            agent_path = '{}{}'.format(generation_path, name)
            print(agent_path)
            private_id = int(name[name.find('i')+1:-5])
            with open(model_path, 'r') as f:
                model =  model_from_json(f.read())
            model.load_weights(agent_path)
            loc = self.initialize_agent(self.world, private_id, teams[private_id], nn_model=model)

    def __str__(self):
        return str(self.world)



def visualize_generation():
    """Visualize simulation of specified generation."""
    # argv[2] = project_name
    if (len(sys.argv) > 2):
        project_name = sys.argv[2]
    else:
        project_name = 'untitled'
    # argv[3] = generation (-1 specifies the last generation of the project)
    if (len(sys.argv) > 3):
        generation = int(sys.argv[3])
    else:
        generation = -1
    # argv[4] = one
    if (len(sys.argv) > 4):
        one = (sys.argv[4] == 'True')
    else:
        one = False
    path = 'out/' + project_name + '/'
    simulation = Simulation(path=path)
    simulation.load_generation(generation=generation)
    simulation.visualize(simulation.world, one=one)

def continue_evolution():
    """Continue runing an evolution from specified generation."""
    # argv[2] = project_name
    if (len(sys.argv) > 2):
        project_name = sys.argv[2]
    else:
        project_name = 'untitled'
    # argv[3] = generation (-1 specifies the last generation of the project)
    if (len(sys.argv) > 3):
        generation = int(sys.argv[3])
    else:
        generation = -1
    # argv[4] = one
    if (len(sys.argv) > 4):
        one = (sys.argv[4] == 'True')
    else:
        one = False
    # argv[5] = draw
    if (len(sys.argv) > 5):
        draw = (sys.argv[5] == 'True')
    else:
        draw = False
    # Create simulation
    path = 'out/' + project_name + '/'
    simulation = Simulation(path=path, draw=draw)
    simulation.load_generation(generation=generation)
    simulation.generation += 1
    simulation.run(simulation.world, one=one)

def run_evolution():
    """Create a simulation and run the evolution."""
    # argv[2] = project_name
    if (len(sys.argv) > 2):
        project_name = sys.argv[2]
    else:
        project_name = 'untitled'
    # argv[3] = one
    if (len(sys.argv) > 3):
        one = (sys.argv[3] == 'True')
    else:
        one = False
    # argv[4] = draw
    if (len(sys.argv) > 4):
        draw = (sys.argv[4] == 'True')
    else:
        draw = False
    # Create simulation
    path = 'out/' + project_name + '/'
    simulation = Simulation(path=path, draw=draw)
    simulation.run(simulation.world, one=one)

def main():
    # argv[1] = command (run | cont | vis)
    # python3 evolution.py run [project_name] [one] [draw]
    # python3 evolution.py cont [project_name] [generation] [one] [draw]
    # python3 evolution.py vis [project_name] [generation] [one]
    if (len(sys.argv) > 1):
        command = sys.argv[1]
    else:
        command = 'run'
    if command == 'run':
        run_evolution()
    elif command == 'cont':
        continue_evolution()
    elif command == 'vis':
        visualize_generation()
    else:
        print('ERROR: unknown command "{}" (run | cont | vis)')


if __name__ == '__main__':
    main()