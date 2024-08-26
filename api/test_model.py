import numpy as np
import sim
import sys 
import matplotlib.pyplot as plt
from environment import NaoEnvironment
from tensorflow.python.keras.models import load_model
from dqn import DQNNetwork

def connect_to_simulation():
    sim.simxFinish(-1)
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()
    print("Connected to CoppeliaSim")
    return clientID

if __name__ == "__main__":
    clientID = connect_to_simulation()
    env = NaoEnvironment(clientID)

    # Load the trained model
    model = load_model('trained_model_episode_50.h5', custom_objects={'DQNNetwork': DQNNetwork})

    num_episodes = 10  # Set the number of episodes you want to run for testing
    total_rewards = []

    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])

        total_reward = 0
        for time in range(50):  # You can adjust the number of time steps
            q_values = model.predict(state)
            action = np.argmax(q_values[0])
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            state = next_state
            if done:
                break

        total_rewards.append(total_reward)
        print(f"Total reward for episode {e + 1}: {total_reward}")

    # Plot the rewards after all episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), total_rewards, marker='o', linestyle='-', color='b')
    plt.title('Total Rewards per Episode during Testing')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

    sim.simxFinish(clientID)