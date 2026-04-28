from pong_game import PongGame
from ai_brain import QLearningAgent
import pygame

def main():
    game = PongGame(visual=True)
    agent = QLearningAgent()
    epsilon = 1.0
    best_streak = 0
    
    print("Training started... Close the window to stop.")

    for episode in range(1, 2001): # Increased to 2000 for better learning
        state = game.reset()
        done = False
        current_streak = 0
        
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            
            if reward == 10: 
                current_streak += 1
            
            # Show every 10th episode so training is faster but still visible
            if episode % 10 == 0:
                stats = {
                    "Episode": episode,
                    "Current Hits": current_streak,
                    "Best Streak": best_streak,
                    "Exploration (ε)": f"{epsilon:.2f}",
                    "Memory Size": len(agent.q_table)
                }
                game.render(stats)
                # pygame.time.delay(2) # Uncomment if it moves too fast to see

        if current_streak > best_streak:
            best_streak = current_streak

        # Decay epsilon (AI explores less and uses its brain more over time)
        epsilon = max(0.01, epsilon * 0.995)

        if episode % 100 == 0:
            print(f"Ep {episode} | Best Streak: {best_streak} | Memory: {len(agent.q_table)}")

    # --- SHOWCASE PHASE ---
    print("Training complete! Watch the AI play perfectly.")
    while True:
        state = game.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state, epsilon=0)
            state, reward, done = game.step(action)
            if reward == 10: score += 1
            game.render({"MODE": "SHOWCASE", "Score": score, "Best": best_streak})
            pygame.time.wait(15)

if __name__ == "__main__":
    main()