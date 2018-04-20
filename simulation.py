from models import World, Agent, Direction


def main():
    # Initialize world
    world = World(5, 5)
    world.agents[(0,0)] = Agent()
    world.agents[(0,0)].move = True
    world.agents[(0,0)].direction = Direction.EAST
    world.agents[(3,3)] = Agent()
    world.agents[(3,3)].move = True
    print '0', world
    # Run simulation
    for t in range(1,10):
        world.update()
        print t, world


if __name__ == '__main__':
    main()