import unittest
from models import World, Agent, Direction


class AgentTestCase(unittest.TestCase):
    def test_agent_initialization(self):
        agent = Agent()
        self.assertEqual(agent.direction, Direction.NORTH,
                            'incorrect initial agent direction')
        self.assertEqual(agent.move, False,
                            'incorrect initial agent move')


class WorldTestCase(unittest.TestCase):
    def test_world_initialization(self):
        world = World(10, 5)
        self.assertEqual((world.width, world.height), (10, 5),
                            'incorrect world dimensions')
        self.assertIsNotNone(world.agents,
                            'agents should be not be None')


class MovementTestCase(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_agent_move_north(self):
        world = World(10, 5)
        agent = Agent()
        agent.direction = Direction.NORTH
        agent.move = True
        world.agents[(0,0)] = agent
        self.assertEqual(world.agents[(0,0)], agent,
                            '(0) world incorrectly set agent position')
        world.update()
        self.assertTrue((0,0) not in world.agents,
                            'agent not moved from previous location')
        self.assertEqual(world.agents[(0,1)], agent,
                            '(1) world incorrectly updated to move agent North')
        world.update()
        self.assertEqual(world.agents[(0,2)], agent,
                            '(2) world incorrectly updated to move agent North')
        world.update()
        self.assertEqual(world.agents[(0,3)], agent,
                            '(3) world incorrectly updated to move agent North')
        world.update()
        self.assertEqual(world.agents[(0,4)], agent,
                            '(4) world incorrectly updated to move agent North')
        world.update()
        self.assertEqual(world.agents[(0,0)], agent,
                            '(5) world incorrectly updated to move agent North (wrap)')
        self.assertEqual(len(world.agents), 1,
                            'World has incorrect number of agents')

    def test_agent_move_south(self):
        world = World(10, 5)
        agent = Agent()
        agent.direction = Direction.SOUTH
        agent.move = True
        world.agents[(0,4)] = agent
        self.assertEqual(world.agents[(0,4)], agent,
                            '(0) world incorrectly set agent position')
        world.update()
        self.assertTrue((0,4) not in world.agents,
                            'agent not moved from previous location')
        self.assertEqual(world.agents[(0,3)], agent,
                            '(1) world incorrectly updated to move agent South')
        world.update()
        self.assertEqual(world.agents[(0,2)], agent,
                            '(2) world incorrectly updated to move agent South')
        world.update()
        self.assertEqual(world.agents[(0,1)], agent,
                            '(3) world incorrectly updated to move agent South')
        world.update()
        self.assertEqual(world.agents[(0,0)], agent,
                            '(4) world incorrectly updated to move agent South')
        world.update()
        self.assertEqual(world.agents[(0,4)], agent,
                            '(5) world incorrectly updated to move agent South (wrap)')
        self.assertEqual(len(world.agents), 1,
                            'World has incorrect number of agents')

    def test_agent_move_east(self):
        world = World(5, 10)
        agent = Agent()
        agent.direction = Direction.EAST
        agent.move = True
        world.agents[(0,0)] = agent
        self.assertEqual(world.agents[(0,0)], agent,
                            '(0) world incorrectly set agent position')
        world.update()
        self.assertTrue((0,0) not in world.agents,
                            'agent not moved from previous location')
        self.assertEqual(world.agents[(1,0)], agent,
                            '(1) world incorrectly updated to move agent East')
        world.update()
        self.assertEqual(world.agents[(2,0)], agent,
                            '(2) world incorrectly updated to move agent East')
        world.update()
        self.assertEqual(world.agents[(3,0)], agent,
                            '(3) world incorrectly updated to move agent East')
        world.update()
        self.assertEqual(world.agents[(4,0)], agent,
                            '(4) world incorrectly updated to move agent East')
        world.update()
        self.assertEqual(world.agents[(0,0)], agent,
                            '(5) world incorrectly updated to move agent East (wrap)')
        self.assertEqual(len(world.agents), 1,
                            'World has incorrect number of agents')

    def test_agent_move_west(self):
        world = World(5, 10)
        agent = Agent()
        agent.direction = Direction.WEST
        agent.move = True
        world.agents[(4,0)] = agent
        self.assertEqual(world.agents[(4,0)], agent,
                            '(0) world incorrectly set agent position')
        world.update()
        self.assertTrue((4,0) not in world.agents,
                            'agent not moved from previous location')
        self.assertEqual(world.agents[(3,0)], agent,
                            '(1) world incorrectly updated to move agent West')
        world.update()
        self.assertEqual(world.agents[(2,0)], agent,
                            '(2) world incorrectly updated to move agent West')
        world.update()
        self.assertEqual(world.agents[(1,0)], agent,
                            '(3) world incorrectly updated to move agent West')
        world.update()
        self.assertEqual(world.agents[(0,0)], agent,
                            '(4) world incorrectly updated to move agent West')
        world.update()
        self.assertEqual(world.agents[(4,0)], agent,
                            '(5) world incorrectly updated to move agent West (wrap)')
        self.assertEqual(len(world.agents), 1,
                            'World has incorrect number of agents')

    def test_agent_collisions(self):
        world = World(10, 5)
        # not moving agent
        agent = Agent()
        # agent moving north
        agentN = Agent()
        agentN.direction = Direction.NORTH
        agentN.move = True
        # agent moving south
        agentS = Agent()
        agentS.direction = Direction.SOUTH
        agentS.move = True
        # agent moving east
        agentE = Agent()
        agentE.direction = Direction.EAST
        agentE.move = True
        # agent moving west
        agentW = Agent()
        agentW.direction = Direction.WEST
        agentW.move = True
        # setup
        world.agents[(2,3)] = agent
        world.agents[(2,0)] = agentN
        world.agents[(2,4)] = agentS
        world.agents[(1,1)] = agentE
        world.agents[(4,1)] = agentW
        # update & test
        world.update()
        self.assertEqual(world.agents[(2,3)], agent,
                            '(1) world incorrectly updated stationary agent')
        self.assertEqual(world.agents[(2,1)], agentN,
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,4)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(1,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(1) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(2,3)], agent,
                            '(2) world incorrectly updated stationary agent')
        self.assertEqual(world.agents[(2,2)], agentN,
                            '(2) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,4)], agentS,
                            '(2) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(2) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(2) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(2,3)], agent,
                            '(3) world incorrectly updated stationary agent')
        self.assertEqual(world.agents[(2,2)], agentN,
                            '(3) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,4)], agentS,
                            '(3) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(3) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(3) world incorrectly updated west agent')
        self.assertEqual(len(world.agents), 5,
                            'World has incorrect number of agents')

    def test_agent_collisions2(self):
        world = World(5, 5)
        # agent moving north
        agentN = Agent()
        agentN.direction = Direction.NORTH
        agentN.move = True
        # agent moving south
        agentS = Agent()
        agentS.direction = Direction.SOUTH
        agentS.move = True
        # agent moving east
        agentE = Agent()
        agentE.direction = Direction.EAST
        agentE.move = True
        # agent moving west
        agentW = Agent()
        agentW.direction = Direction.WEST
        agentW.move = True
        # setup
        world.agents[(2,0)] = agentN
        world.agents[(2,4)] = agentS
        world.agents[(1,1)] = agentE
        world.agents[(3,1)] = agentW
        # update & test
        world.update()
        self.assertEqual(world.agents[(2,1)], agentN,
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,3)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(1,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(1) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(2,2)], agentN,
                            '(2) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,3)], agentS,
                            '(2) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(2) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(2) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(2,2)], agentN,
                            '(3) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,3)], agentS,
                            '(3) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(3) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(3) world incorrectly updated west agent')
        self.assertEqual(len(world.agents), 4,
                            'World has incorrect number of agents')

    def test_agent_gridlock(self):
        world = World(2, 2)
        # agent moving north
        agentN = Agent()
        agentN.direction = Direction.NORTH
        agentN.move = True
        # agent moving south
        agentS = Agent()
        agentS.direction = Direction.SOUTH
        agentS.move = True
        # agent moving east
        agentE = Agent()
        agentE.direction = Direction.EAST
        agentE.move = True
        # agent moving west
        agentW = Agent()
        agentW.direction = Direction.WEST
        agentW.move = True
        # setup
        world.agents[(0,0)] = agentN
        world.agents[(1,1)] = agentS
        world.agents[(0,1)] = agentE
        world.agents[(1,0)] = agentW
        # update & test
        world.update()
        self.assertEqual(world.agents[(0,0)], agentN,
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(1,1)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(0,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(1,0)], agentW,
                            '(1) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(0,0)], agentN,
                            '(2) world incorrectly updated north agent')
        self.assertEqual(world.agents[(1,1)], agentS,
                            '(2) world incorrectly updated south agent')
        self.assertEqual(world.agents[(0,1)], agentE,
                            '(2) world incorrectly updated east agent')
        self.assertEqual(world.agents[(1,0)], agentW,
                            '(2) world incorrectly updated west agent')
        self.assertEqual(len(world.agents), 4,
                            'World has incorrect number of agents')

    def test_agent_gridlock2(self):
        world = World(3, 3)
        # agent moving north
        agentN = Agent()
        agentN.direction = Direction.NORTH
        agentN.move = True
        # agent moving north2
        agentN2 = Agent()
        agentN2.direction = Direction.NORTH
        agentN2.move = True
        # agent moving north3
        agentN3 = Agent()
        agentN3.direction = Direction.NORTH
        agentN3.move = True
        # agent moving south
        agentS = Agent()
        agentS.direction = Direction.SOUTH
        agentS.move = True
        # agent moving east
        agentE = Agent()
        agentE.direction = Direction.EAST
        agentE.move = True
        # agent moving west
        agentW = Agent()
        agentW.direction = Direction.WEST
        agentW.move = True
        # agent moving west2
        agentW2 = Agent()
        agentW2.direction = Direction.WEST
        agentW2.move = True
        # setup
        world.agents[(1,1)] = agentN
        world.agents[(0,0)] = agentN2
        world.agents[(2,0)] = agentN3
        world.agents[(0,2)] = agentS
        world.agents[(0,1)] = agentE
        world.agents[(1,2)] = agentW
        world.agents[(2,1)] = agentW2
        # update & test
        world.update()
        self.assertEqual(world.agents[(1,1)], agentN,
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(0,0)], agentN2,
                            '(1) world incorrectly updated north2 agent')
        self.assertEqual(world.agents[(2,0)], agentN3,
                            '(1) world incorrectly updated north3 agent')
        self.assertEqual(world.agents[(0,2)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(0,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(1,2)], agentW,
                            '(1) world incorrectly updated west agent')
        self.assertEqual(world.agents[(2,1)], agentW2,
                            '(1) world incorrectly updated west2 agent')
        world.update()
        self.assertEqual(world.agents[(1,1)], agentN,
                            '(2) world incorrectly updated north agent')
        self.assertEqual(world.agents[(0,0)], agentN2,
                            '(2) world incorrectly updated north2 agent')
        self.assertEqual(world.agents[(2,0)], agentN3,
                            '(2) world incorrectly updated north3 agent')
        self.assertEqual(world.agents[(0,2)], agentS,
                            '(2) world incorrectly updated south agent')
        self.assertEqual(world.agents[(0,1)], agentE,
                            '(2) world incorrectly updated east agent')
        self.assertEqual(world.agents[(1,2)], agentW,
                            '(2) world incorrectly updated west agent')
        self.assertEqual(world.agents[(2,1)], agentW2,
                            '(2) world incorrectly updated west2 agent')
        self.assertEqual(len(world.agents), 7,
                            'World has incorrect number of agents')

    def test_agent_gridlock_wrap(self):
        world = World(3, 3)
        # agent moving north
        agentN = Agent()
        agentN.direction = Direction.NORTH
        agentN.move = True
        # agent moving south
        agentS = Agent()
        agentS.direction = Direction.SOUTH
        agentS.move = True
        # agent moving east
        agentE = Agent()
        agentE.direction = Direction.EAST
        agentE.move = True
        # agent moving west
        agentW = Agent()
        agentW.direction = Direction.WEST
        agentW.move = True
        # setup
        world.agents[(1,2)] = agentN
        world.agents[(1,0)] = agentS
        world.agents[(2,1)] = agentE
        world.agents[(0,1)] = agentW
        # update & test
        world.update()
        self.assertEqual(world.agents[(1,2)], agentN,
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(1,0)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(0,1)], agentW,
                            '(1) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(1,2)], agentN,
                            '(2) world incorrectly updated north agent')
        self.assertEqual(world.agents[(1,0)], agentS,
                            '(2) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(2) world incorrectly updated east agent')
        self.assertEqual(world.agents[(0,1)], agentW,
                            '(2) world incorrectly updated west agent')
        self.assertEqual(len(world.agents), 4,
                            'World has incorrect number of agents')


if __name__ == '__main__':
    unittest.main()
