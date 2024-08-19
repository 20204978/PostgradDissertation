import sim
import sys

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
    
