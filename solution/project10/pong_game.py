import pygame

class PongGame:
    def __init__(self, visual=True):
        self.width, self.height = 400, 300
        self.visual = visual
        if self.visual:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Pong Training")
            self.font = pygame.font.SysFont("Arial", 18) # For the HUD
        self.reset()

    def reset(self):
        self.paddle_y = self.height // 2
        self.ball_x, self.ball_y = self.width // 2, self.height // 2
        self.ball_dx, self.ball_dy = 5, 5
        return self.get_state()

    def step(self, action):
        if action == 0 and self.paddle_y > 0:
            self.paddle_y -= 10
        elif action == 1 and self.paddle_y < self.height - 60:
            self.paddle_y += 10

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        reward = 0
        done = False

        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dy *= -1
        
        if self.ball_x >= self.width:
            if self.paddle_y < self.ball_y < self.paddle_y + 60:
                self.ball_dx *= -1.05 
                reward = 10 # Hit!
            else:
                reward = -10 # Miss!
                done = True 
        elif self.ball_x <= 0:
            self.ball_dx *= -1
            
        return self.get_state(), reward, done

    def get_state(self):
        return (int(self.ball_x // 40), int(self.ball_y // 40), int(self.paddle_y // 40))

    def render(self, stats=None):
        if not self.visual: return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()

        self.screen.fill((20, 20, 30)) # Darker background
        
        # Draw Paddle and Ball
        pygame.draw.rect(self.screen, (255, 255, 255), (380, self.paddle_y, 10, 60))
        pygame.draw.circle(self.screen, (0, 255, 127), (int(self.ball_x), int(self.ball_y)), 6)

        # --- DRAW HUD ---
        if stats:
            y_offset = 10
            for key, value in stats.items():
                text_surface = self.font.render(f"{key}: {value}", True, (200, 200, 200))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 20

        pygame.display.flip()