import random
import numpy as np
import pickle 

class QLearningAgent:
    def __init__(self):
        # Initialize an empty dictionary to act as the AI's memory (Q-Table)
        self.Q_Table = {}
        

    def get_q_values(self, state):
        # Check if the current state is already in our memory
        # If the state is new, create a list of three zeros for our possible moves: Up, Down, Stay
        # Return the memory values for this state
        if state not in self.Q_Table:
            self.Q_Table[state] = [0.0,0.0,0.0]
        return self.Q_Table[state]

    def choose_action(self, state, epsilon):
        # Generate a random number to decide between exploring and exploiting
        # If the random number is less than epsilon, pick a random move (0, 1, or 2)
        # Otherwise, look at the memory and pick the move with the highest value using argmax
        if random.random()<epsilon:
            return random.randint(0,2)
        else:
            return np.argmax(self.get_q_values(state))
        

    def learn(self, state, action, reward, next_state):
        # Look up the current knowledge (Q-values) for the current state
        # Look up the knowledge for the state the ball moved into next
        
        # Define the learning rate (how much we trust new info)
        # Define the discount factor (how much we care about future rewards)
        
        # Find the best possible value we could get in the next state
        # Use the Q-Learning formula to update the value of the action we just took
        # This formula balances what we knew before with the new reward we just received
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        alpha, gamma = 0.2, 0.9
        best_future_q = max(next_q_values)
        q_values[action]+=alpha*(reward+gamma*best_future_q-q_values[action])
        
    
    def save_model(self, filename="pong_ai.pkl"):
        # Open a file in write-binary mode
        # Use the pickle library to dump our memory dictionary into the file
        # Print a confirmation message to the console
        pass

    def load_model(self, filename="pong_ai.pkl"):
        # Use a try block in case the file doesn't exist yet
        # Open the file in read-binary mode
        # Load the data back into our memory dictionary
        # Print a success message
        # If the file is missing, catch the error and print a warning
        pass