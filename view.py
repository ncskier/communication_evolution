from models import World, Agent, Direction
import pygame

class View:
    """View to display world."""

    def __init__(self, world):
        # Initialize pygame module
        pygame.init()
        # Set screen size of pygame window
        self.screen = pygame.display.set_mode((640, 480))
        # Create background
        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill((255, 255, 255))
        self.background = self.background.convert()   # convert surface to make blitting faster
        self.screen.blit(self.background, (0, 0))
        # Create grid
        self.cell_size = self.calculate_cell_size(world)
        # Update pygame display
        pygame.display.flip()

    def draw(self, world):
        """Draw [world]."""
        # Erase screen with background
        self.screen.blit(self.background, (0, 0))
        # Draw agents
        for grid_loc in world.agents:
            draw_x, draw_y = self.grid_to_screen(grid_loc, world)
            width, height = self.cell_size
            rect = (draw_x, draw_y, width, height)
            color = (0, 255, 0)
            pygame.draw.rect(self.screen, color, rect)
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

    def save(self, path):
        """Save image of view to a file."""
        pygame.image.save(self.screen, path)
