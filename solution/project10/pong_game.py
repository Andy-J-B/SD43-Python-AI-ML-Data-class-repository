import pygame
import random

class PongGame:
    def __init__(self, visual=True):
        self.width, self.height = 400, 300
        self.visual = visual
        if self.visual:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Pong Training")
            self.font = pygame.font.SysFont("Arial", 18)
        self.reset()

    def reset(self):
        # 1. Reset paddle to middle
        self.paddle_y = self.height // 2
        
        # 2. Spawn ball in the center horizontally
        self.ball_x = self.width // 2
        
        # 3. Randomize vertical start (staying away from the very edges)
        self.ball_y = random.randint(50, self.height - 50)
        
        # 4. Randomize direction
        # ball_dx: starts moving left or right randomly
        # ball_dy: starts moving up or down at varying angles
        self.ball_dx = random.choice([-5, 5])
        self.ball_dy = random.choice([-6, -4, 4, 6])
        
        return self.get_state()

    def step(self, action):
        # (Rest of the step logic remains the same)
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
                reward = 10 
            else:
                reward = -10 
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
        self.screen.fill((20, 20, 30))
        pygame.draw.rect(self.screen, (255, 255, 255), (380, self.paddle_y, 10, 60))
        pygame.draw.circle(self.screen, (0, 255, 127), (int(self.ball_x), int(self.ball_y)), 6)
        if stats:
            y_offset = 10
            for key, value in stats.items():
                text_surface = self.font.render(f"{key}: {value}", True, (200, 200, 200))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 20
        pygame.display.flip()