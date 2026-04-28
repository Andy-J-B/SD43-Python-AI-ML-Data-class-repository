import random
import numpy as np
import pickle 

class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.get_q_values(state))

    def learn(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        
        alpha = 0.2  # Learning Rate
        gamma = 0.9  # Discount Factor
        
        best_future_q = max(next_q_values)
        q_values[action] += alpha * (reward + gamma * best_future_q - q_values[action])
    
    def save_model(self, filename="pong_ai.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename="pong_ai.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print("No saved model found. Starting with a fresh brain.")