import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size    # e.g. 10 features
        self.action_size = action_size  # Buy, Sell, Hold

        # Hyperparameters
        self.gamma = 0.95               # Discount factor
        self.epsilon = 1.0              # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)

        # Model
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))  # Output: Q-values for each action
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                q_next = self.model.predict(next_state[np.newaxis, :], verbose=0)[0]
                target = reward + self.gamma * np.amax(q_next)

            q_values = self.model.predict(state[np.newaxis, :], verbose=0)
            q_values[0][action] = target

            self.model.fit(state[np.newaxis, :], q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
