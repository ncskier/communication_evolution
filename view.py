from models import World, Agent, Direction
import pygame
import math
import os

class View:
    """View to display world."""

    def __init__(self, world):
        # Initialize pygame module
        pygame.init()
        # Set screen size of pygame window
        self.screen = pygame.display.set_mode((640, 480))
        # Create grid
        self.cell_size = self.calculate_cell_size(world)
        # Create background
        self.init_background(world)
        # Update pygame display
        pygame.display.flip()

    def init_background(self, world):
        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill((255, 255, 255))
        self.background = self.background.convert()   # convert surface to make blitting faster
        grid_color = (0, 0, 0)
        for x in range(world.width-1):
            start_pos = self.grid_to_screen((x+1, -1), world)
            end_pos = self.grid_to_screen((x+1, world.height-1), world)
            pygame.draw.line(self.background, grid_color, start_pos, end_pos)
        for y in range(world.height-1):
            start_pos = self.grid_to_screen((0, y), world)
            end_pos = self.grid_to_screen((world.width, y), world)
            pygame.draw.line(self.background, grid_color, start_pos, end_pos)

    def draw(self, world):
        """Draw [world]."""
        # Erase screen with background
        self.screen.blit(self.background, (0, 0))
        # Draw agents
        for grid_loc in world.agents:
            agent = world.agents[grid_loc]
            rect = self.agent_rect(grid_loc, world)
            # Draw agent triangle
            color = self.agent_color(agent.private_id, world)
            pointlist = self.agent_pointlist(rect, agent.direction)
            pygame.draw.polygon(self.screen, color, pointlist)
            # Draw agent triangle border
            border_color = (0, 0, 0)
            border_width = math.ceil(rect[2]*0.05)
            if agent.move and agent.direction == agent.correct_direction:
                border_color = (255, 0, 0)
            pygame.draw.polygon(self.screen, border_color, pointlist, border_width)
        # Commit frame to screen
        pygame.display.flip()

    def calculate_cell_size(self, world):
        """Calculate cell size in the grid."""
        screen_width, screen_height = self.screen.get_size()
        width = screen_width / world.width
        height = screen_height / world.height
        return (width, height)

    def grid_to_screen(self, grid_loc, world):
        """Convert grid location to screen location."""
        grid_x, grid_y = grid_loc
        cell_width, cell_height = self.cell_size
        screen_x = grid_x * cell_width
        screen_y = (world.height-grid_y-1) * cell_height
        return (screen_x, screen_y)

    def agent_color(self, id, world):
        """Calculate agent color from its [id]."""
        max_id = len(world.agents) - 1
        min_color = 0
        max_color = 255     # really 255 max
        base = max_color - min_color
        max_num = pow(max_color-min_color, 3) - 1
        num = (id / max_id) * max_num
        r = min_color + (num % base)
        g = min_color + ((num/base) % base)
        b = min_color + ((num/base/base) % base)
        return (r, g, b)

    def agent_rect(self, grid_loc, world):
        """Calculate rect for drawing agent at [grid_loc]."""
        # Make squares
        draw_x, draw_y = self.grid_to_screen(grid_loc, world)
        width, height = self.cell_size
        if width > height:
            length = height
            x = draw_x + (width-height)/2.0
            y = draw_y
        else:
            length = width
            x = draw_x
            y = draw_y + (height-width)/2.0
        # Add padding
        padding = length*0.1
        x += padding
        y += padding
        length -= padding*2.0
        return (x, y, length, length)

    def agent_pointlist(self, rect, direction):
        """Calculate polygon pointlist given agent's [rect] and [direction]."""
        x, y, width, height = rect
        pointlist = []
        if direction == Direction.NORTH:
            # North
            pointlist.append((x, y+height))
            pointlist.append((x+width, y+height))
            pointlist.append((x + width/2.0, y))
        elif direction == Direction.SOUTH:
            # South
            pointlist.append((x, y))
            pointlist.append((x + width/2.0, y+height))
            pointlist.append((x+width, y))
        elif direction == Direction.EAST:
            # East
            pointlist.append((x, y))
            pointlist.append((x, y+height))
            pointlist.append((x+width, y + height/2.0))
        else:
            # West
            pointlist.append((x, y + height/2.0))
            pointlist.append((x+width, y+height))
            pointlist.append((x+width, y))
        return pointlist

    def save(self, path):
        """Save image of view to a file."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        pygame.image.save(self.screen, path)
