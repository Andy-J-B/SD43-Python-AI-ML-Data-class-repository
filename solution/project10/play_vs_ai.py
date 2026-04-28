import pygame
from pong_game import PongGame
from ai_brain import QLearningAgent

class VersusGame(PongGame):
    def __init__(self):
        super().__init__(visual=True)
        self.human_y = self.height // 2
        self.human_score = 0
        self.ai_score = 0

    def play_step(self, ai_action, human_action):
        # AI Paddle (Right)
        if ai_action == 0 and self.paddle_y > 0: self.paddle_y -= 10
        elif ai_action == 1 and self.paddle_y < self.height - 60: self.paddle_y += 10

        # Human Paddle (Left)
        if human_action == "UP" and self.human_y > 0: self.human_y -= 10
        elif human_action == "DOWN" and self.human_y < self.height - 60: self.human_y += 10

        # Ball Physics
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Bounce Top/Bottom
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dy *= -1

        # Bounce Right (AI Paddle)
        if self.ball_x >= self.width - 20:
            if self.paddle_y < self.ball_y < self.paddle_y + 60:
                self.ball_dx *= -1.05
            else:
                self.human_score += 1
                self.reset()

        # Bounce Left (Human Paddle)
        if self.ball_x <= 15:
            if self.human_y < self.ball_y < self.human_y + 60:
                self.ball_dx *= -1.05
            else:
                self.ai_score += 1
                self.reset()

    def render_versus(self):
        self.screen.fill((30, 30, 30))
        # Draw Human Paddle
        pygame.draw.rect(self.screen, (0, 150, 255), (10, self.human_y, 10, 60))
        # Draw AI Paddle
        pygame.draw.rect(self.screen, (255, 50, 50), (380, self.paddle_y, 10, 60))
        # Draw Ball
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), 6)
        
        # Draw Scores
        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render(f"YOU: {self.human_score}  |  AI: {self.ai_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.width // 2 - 70, 10))
        
        pygame.display.flip()

def main():
    agent = QLearningAgent()
    agent.load_model("pong_ai.pkl") # Load the training result
    game = VersusGame()
    clock = pygame.time.Clock()

    while True:
        human_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return

        # Get Human Keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: human_action = "UP"
        if keys[pygame.K_s]: human_action = "DOWN"

        # Get AI Action
        state = game.get_state()
        ai_action = agent.choose_action(state, epsilon=0) # No randomness

        game.play_step(ai_action, human_action)
        game.render_versus()
        clock.tick(60) # Smooth 60 FPS

if __name__ == "__main__":
    main()