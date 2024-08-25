import sim
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from environment import NaoEnvironment  # Import environment class

def connect_to_simulation():
    sim.simxFinish(-1)  # Close any open connections just in case
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    
    if clientID == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()

    print("Connected to CoppeliaSim")
    return clientID

if __name__ == "__main__":
    # Connect to the simulation
    clientID = connect_to_simulation()
    env = NaoEnvironment(clientID)

    # Load the trained model
    with open('trained_model_episode_150.pkl', 'rb') as f:
        agent = pickle.load(f)

    state_size = env.state_size
    action_size = env.action_size

    total_rewards = []

    # Set epsilon to 0 to ensure no exploration (greedy policy)
    agent.epsilon = 0.0

    for e in range(10):  # Test on 10 episodes
        print(f"Starting test episode {e+1}")
        
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(100):  # Max time steps per episode
            action = agent.act(state)  # Get action from the trained model
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            
            if done:
                print(f"Episode {e+1} finished after {time} time steps with reward {total_reward}")
                break

        total_rewards.append(total_reward)
        print(f"Total reward for episode {e+1}: {total_reward}")

    # Calculate average reward
    average_reward = np.mean(total_rewards)
    print(f"Average reward over 10 episodes: {average_reward}")

    # Plot the rewards for each episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(total_rewards) + 1), total_rewards, marker='o', linestyle='-', color='b')
    plt.title('Total Rewards per Episode during Testing')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

    sim.simxFinish(clientID)  # Close the connection to the simulation
