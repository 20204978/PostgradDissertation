# Imports
import numpy as np
import sim
import random
import tensorflow as tf
from collections import deque
from environment import NaoEnvironment

# Define the DQN network architecture
class DQNNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, **kwargs):
        super(DQNNetwork, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)
    
    # Add serialisation methods
    def get_config(self):
        config = super(DQNNetwork, self).get_config()
        config.update({
            "state_size": self.fc1.input_shape[-1],
            "action_size": self.fc3.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # initial Exploration rate
        self.epsilon_min = 0.01 # minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate per ep
        self.decay_interval = 10 # Apply decay every 10 epa
        self.learning_rate = 0.001
        self.episode_counter = 0  # Initialise episode counter
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.update_target_model()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.MeanSquaredError())

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore 
        q_values = self.model.predict(state)
        return np.argmax(q_values[0]) # Exploit

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Increment the episode counter
        self.episode_counter += 1
        # Apply epsilon decay every 'decay_interval' episodes
        if self.episode_counter % self.decay_interval == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay  # Decay epsilon



