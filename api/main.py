import sim
import time
import sys
import numpy as np
import logging
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import summary
from environment import NaoEnvironment
from dqn import DQNAgent  

# Configure logging
logging.basicConfig(filename='training_log_new3.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def connect_to_simulation():
    sim.simxFinish(-1)  # Close any open connections just in case
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    
    if clientID == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()

    print("Connected to CoppeliaSim")
    return clientID

def plot_rewards(rewards, episode):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title(f'Training Progress - Episode {episode}')
    plt.grid(True)
    plt.savefig(f'rewards_plot_episode_{episode}.png')
    plt.close()

if __name__ == "__main__":
    clientID = connect_to_simulation()
    env = NaoEnvironment(clientID)

    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    rewards = []

    best_reward = -np.inf
    patience = 100  # Early stopping patience
    patience_counter = 0

    # TensorFlow setup
    log_dir = "logs/fit/" + "training_" + str(int(time.time()))
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{episode}")
    checkpoint = tf.train.Checkpoint(model=agent.model)

    for e in range(1000):  # Number of eps
        logging.info(f"Starting episode {e+1}")
        print(f"Starting episode {e+1}")  # Also print to terminal
        
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(100):  # Time steps per ep
            action = agent.act(state)
            next_state, reward, done = env.step(action)
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

        # Log rewards to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('reward', total_reward, step=e)
            tf.summary.scalar('epsilon', agent.epsilon, step=e)

        # Check for early stopping
        if total_reward > best_reward:
            best_reward = total_reward
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break

        # Periodically save the model during training (every 50 eps)
        if e % 50 == 0:
            with open(f'trained_model_episode_{e}.pkl', 'wb') as f:
                pickle.dump(agent, f)
            logging.info(f"Model saved after episode {e}")
            print(f"Model saved after episode {e}")

            # Plot and save rewards
            plot_rewards(rewards, e + 1)

     # Final save after all episodes
    with open('trained_model_final.pkl', 'wb') as f:
        pickle.dump(agent, f)
    logging.info("Final model saved as 'trained_model_final.pkl'")
    print("Final model saved as 'trained_model_final.pkl'")
    
    # Save rewards for later analysis
    np.save('rewards.npy', rewards)
    logging.info("Training finished, rewards saved to 'rewards.npy'")
    print("Training finished, rewards saved to 'rewards.npy'")
    
    sim.simxFinish(clientID)

