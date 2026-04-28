from pong_game import PongGame
from ai_brain import QLearningAgent
import pygame

def main():
    # Create the game environment and the AI agent
    # Set the starting exploration rate (epsilon) to 1.0
    # Keep track of the best hit-streak we've seen so far
    
    # Start a loop that runs for 2000 training episodes
        # Reset the game to start a new round
        # Set the 'done' status to False and reset the current hit counter
        
        # Create a nested loop that runs until the game is over
            # Ask the agent to choose an action based on the current state and epsilon
            # Tell the game to perform that action and get back the result
            # Tell the agent to learn from the transition (State -> Action -> Reward -> Next State)
            # Update our current state variable to be the new state
            
            # If the reward was positive, increment our current hit counter
            
            # Every 10 episodes, render the game so we can watch the progress
            # Pass a dictionary of stats to the render function to show on screen
            
        # After a round ends, check if the current streak is higher than our record
        
        # Reduce epsilon slightly so the AI explores less and relies on memory more
        
        # Every 100 episodes, print a progress report to the console

    # Once training is finished, save the AI's brain to a file

    # --- SHOWCASE PHASE ---
    # Enter an infinite loop to watch the AI play
        # Reset the game
        # While the game isn't over, have the AI choose moves with zero randomness
        # Update the game and render the "Showcase" mode with a small delay
        pass

if __name__ == "__main__":
    main()