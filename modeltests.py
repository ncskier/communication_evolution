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
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,3)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(1) world incorrectly updated west agent')
        world.update()
        self.assertEqual(world.agents[(2,2)], agentN,
                            '(1) world incorrectly updated north agent')
        self.assertEqual(world.agents[(2,3)], agentS,
                            '(1) world incorrectly updated south agent')
        self.assertEqual(world.agents[(2,1)], agentE,
                            '(1) world incorrectly updated east agent')
        self.assertEqual(world.agents[(3,1)], agentW,
                            '(1) world incorrectly updated west agent')


if __name__ == '__main__':
    unittest.main()
