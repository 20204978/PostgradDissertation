import sim
import sys
import numpy as np
from environment import NaoEnvironment
from dqn import DQNAgent  

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

    for e in range(1000):  # Number of eps
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(10):  # Time steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"Episode {e+1}/{1000}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    
    sim.simxFinish(clientID)

