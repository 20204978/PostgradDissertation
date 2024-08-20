import sim
import sys
import numpy as np
import logging
from environment import NaoEnvironment
from dqn import DQNAgent  

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def connect_to_simulation():
    sim.simxFinish(-1)  # Close any open connections just in case
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    
    if clientID == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()

    print("Connected to CoppeliaSim")
    return clientID

if __name__ == "__main__":
    clientID = connect_to_simulation()
    env = NaoEnvironment(clientID)

    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    rewards = []

    for e in range(1000):  # Number of eps
        logging.info(f"Starting episode {e+1}")
        print(f"Starting episode {e+1}")  # Also print to terminal
        
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(10):  # Time steps per ep
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Print and log progress every 100 time steps
            if time % 100 == 0:
                logging.info(f"Episode {e+1}, Time Step {time}: Reward {reward}, Epsilon {agent.epsilon:.2}")
                print(f"Episode {e+1}, Time Step {time}: Reward {reward}, Epsilon {agent.epsilon:.2}")

            if done:
                agent.update_target_model()
                logging.info(f"Episode {e+1} finished after {time} time steps with final reward {reward}")
                print(f"Episode {e+1} finished after {time} time steps with final reward {reward}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        rewards.append(total_reward)
        logging.info(f"Total reward for episode {e+1}: {total_reward}")
        print(f"Total reward for episode {e+1}: {total_reward}")
    
    # Save rewards for later analysis
    np.save('rewards.npy', rewards)
    logging.info("Training finished, rewards saved to 'rewards.npy'")
    print("Training finished, rewards saved to 'rewards.npy'")
    
    sim.simxFinish(clientID)

