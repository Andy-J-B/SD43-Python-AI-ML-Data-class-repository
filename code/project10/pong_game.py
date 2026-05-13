import pygame
import random

class PongGame:
    def __init__(self, visual=True):
        # Set the screen width to 400 and height to 300
        # Store whether we want to show the graphics or train silently
        # If visual is enabled, initialize pygame and create the window
        # Set the window title and prepare a font for the UI
        # Call the reset function to set up the first game
        self.width, self.height = 400, 300
        self.visual = visual
        if self.visual == True:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Pong Training")
            self.font = pygame.font.SysFont("Arial", 18)
        self.reset()

    def reset(self):
        # Place the paddle in the vertical center of the screen
        # Place the ball in the horizontal center of the screen
        # Set the ball's vertical start to a random spot, avoiding the very top/bottom
        # Give the ball a random starting direction (Left/Right and Up/Down)
        # Return the current state of the game so the AI knows where it is starting
        pass

    def step(self, action):
        # If the action is 0 and the paddle isn't at the top, move it up
        # If the action is 1 and the paddle isn't at the bottom, move it down

        # Update the ball's horizontal and vertical positions using its speed variables

        # Initialize the reward for this step to zero
        # Set the 'done' flag to False
        
        # Check if the ball hit the top or bottom wall
        # If it did, reverse its vertical direction
        
        # Check if the ball reached the right side (where the paddle is)
        # Check if the ball's height is within the paddle's range
        # If hit: reverse the ball's horizontal speed, speed it up slightly, and give a positive reward
        # If missed: set the reward to a negative value and end the game
        
        # If the ball hit the left wall, just bounce it back
            
        # Return the new state, the reward earned, and whether the game is over
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
            if self.paddle_y< self.ball_y < self.paddle_y+60:
                self.ball_x *= -1.05
                reward = 10
            else:
                reward = -10
                done = True
        elif self.ball_x <= 0:
            self.ball_dx *= -1
        return self.get_state(), reward, done

        

    def get_state(self):
        # Convert the continuous ball and paddle positions into a simplified grid
        # Divide the X and Y coordinates by a factor (like 40) to make the "map" smaller
        # Return these simplified coordinates as a tuple for the AI's memory keys
        return (int(self.ball_x // 40), int(self.ball_y // 40), int(self.paddle_y // 40))


    def render(self, stats=None):
        # If visual mode is off, exit the function
        # Check for the window close event to prevent the program from crashing
        # Fill the screen with a dark background color
        # Draw the paddle as a white rectangle
        # Draw the ball as a bright green circle
        # If stats are provided, loop through them and draw the text on the screen
        # Refresh the display to show the new frame
        if not self.visual: return
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
        self.screen.fill((20, 20, 30))
        pygame.draw.rect(self.screen,(255,255,255),(380,self.paddle_y,10,60))
        pygame.draw.circle(self.screen,(0,255,127),(int(self.ball_x),int(self.ball_y)),6)
        if stats:
            y_offset = 10
            for key,value in stats.items():
                text = self.font.render(f"{key}: {value}", True, (255, 255, 255))
                self.screen.blit(text, (10, y_offset))
                y_offset += 20
        pygame.display.flip()

