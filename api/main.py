import sim
import sys
from environment import NaoEnvironment

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

    state = env.reset()
    print("Initial state:", state)

    action = np.zeros(env.action_size)  # Example action
    next_state, reward, done = env.step(action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    sim.simxFinish(clientID)
