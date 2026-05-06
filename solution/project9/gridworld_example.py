import pygame
import numpy as np
import time
import random

# --- Configuration ---
ROWS, COLS = 5, 5
CELL_SIZE = 100
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
FPS = 60

# Colors
COLOR_ICE = (200, 230, 255)
COLOR_HOLE = (30, 30, 30)
COLOR_GOAL = (50, 205, 50)
COLOR_AGENT = (255, 69, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_GRID = (150, 150, 150)

# Game Rules
START = (0, 0)
GOAL = (4, 4)
HOLES = {(1, 2), (2, 3), (3, 1), (3, 3)}
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# ==================================================================
# 1️⃣ Core Logic (Environment & Agent)
# ==================================================================

class GridWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = START
        return self.pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc

        # Boundary check
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            self.pos = (nr, nc)

        if self.pos == GOAL:
            return self.pos, 1.0, True
        elif self.pos in HOLES:
            return self.pos, -1.0, True
        else:
            return self.pos, -0.01, False

class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.95, eps=0.2):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.Q = np.zeros((ROWS * COLS, len(ACTIONS)))

    def _state_to_index(self, state):
        return state[0] * COLS + state[1]

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, 3)
        s_idx = self._state_to_index(state)
        # Break ties randomly for max values
        max_val = np.max(self.Q[s_idx])
        actions_with_max = np.where(self.Q[s_idx] == max_val)[0]
        return np.random.choice(actions_with_max)

    def update(self, state, action, reward, next_state, done):
        s_idx = self._state_to_index(state)
        next_s_idx = self._state_to_index(next_state)
        
        best_next_q = 0 if done else np.max(self.Q[next_s_idx])
        # Q-Learning Formula: Q(s,a) = Q(s,a) + lr * (reward + gamma * maxQ(s',a') - Q(s,a))
        self.Q[s_idx, action] += self.lr * (reward + self.gamma * best_next_q - self.Q[s_idx, action])

# ==================================================================
# 2️⃣ UI & Visualization
# ==================================================================

def draw_grid(screen, agent_pos, agent_color=COLOR_AGENT):
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # Draw Cell Type
            if (r, c) == GOAL:
                pygame.draw.rect(screen, COLOR_GOAL, rect)
            elif (r, c) in HOLES:
                pygame.draw.rect(screen, COLOR_HOLE, rect)
            else:
                pygame.draw.rect(screen, COLOR_ICE, rect)
            
            # Draw Grid Lines
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)

    # Draw Agent
    agent_center = (agent_pos[1] * CELL_SIZE + CELL_SIZE // 2, 
                    agent_pos[0] * CELL_SIZE + CELL_SIZE // 2)
    pygame.draw.circle(screen, agent_color, agent_center, CELL_SIZE // 3)

def run_visual_demo():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Frozen Lake UI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24, bold=True)

    env = GridWorld()
    agent = QLearningAgent()

    # --- Step 1: Fast Training ---
    print("Training Agent... Please wait.")
    for _ in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

    # --- Step 2: Visual Playback Loop ---
    running = True
    while running:
        state = env.reset()
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Greedy Choice
            action = agent.select_action(state, greedy=True)
            state, reward, done = env.step(action)

            # Rendering
            screen.fill((0, 0, 0))
            draw_grid(screen, state)
            
            if done:
                msg = "SUCCESS!" if state == GOAL else "FELL IN HOLE!"
                color = (255, 255, 255) if state == GOAL else (255, 0, 0)
                text = font.render(msg, True, color)
                screen.blit(text, (WIDTH // 2 - 50, HEIGHT // 2))
                
            pygame.display.flip()
            clock.tick(5) # Slow down for visibility (5 steps per second)
            
            if done:
                time.sleep(1) # Pause at the end of an episode

    pygame.quit()

if __name__ == "__main__":
    run_visual_demo()