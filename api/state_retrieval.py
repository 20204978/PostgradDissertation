import sim
import numpy as np
import sys

def connect_to_simulation():
    sim.simxFinish(-1)  # Close any open connections just in case
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    
    if clientID == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()

    print("Connected to CoppeliaSim")
    return clientID

def get_joint_states(clientID, joint_names):
    state = []
    for joint in joint_names:
        res, handle = sim.simxGetObjectHandle(clientID, joint, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            # Get joint angle
            res, position = sim.simxGetJointPosition(clientID, handle, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                state.append(position)
            else:
                print(f"Failed to get joint position for {joint}")
            # Get joint velocity
            res, velocity = sim.simxGetObjectFloatParameter(clientID, handle, 2012, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                state.append(velocity)
            else:
                print(f"Failed to get joint velocity for {joint}")
        else:
            print(f"Failed to get joint handle for {joint}")
    return np.array(state)

if __name__ == "__main__":
    joint_names = ['/NAO/RHipYawPitch', '/NAO/RHipRoll', '/NAO/RHipPitch',
                   '/NAO/RKneePitch', '/NAO/RAnklePitch', '/NAO/RAnkleRoll',
                   '/NAO/LHipYawPitch', '/NAO/LHipRoll', '/NAO/LHipPitch',
                   '/NAO/LKneePitch', '/NAO/LAnklePitch', '/NAO/LAnkleRoll',
                   '/NAO/RShoulderPitch', '/NAO/RShoulderRoll', '/NAO/RElbowYaw',
                   '/NAO/RElbowRoll', '/NAO/RWristYaw', '/NAO/RThumbBase', '/NAO/RThumbBase/joint',
                   '/NAO/RRFingerBase', '/NAO/RRFingerBase/joint', '/NAO/RRFingerBase/joint/Cuboid/joint',
                   '/NAO/RLFingerBase', '/NAO/RLFingerBase/joint', '/NAO/RLFingerBase/joint/Cuboid/joint',
                   '/NAO/LShoulderPitch', '/NAO/LShoulderRoll', '/NAO/LElbowYaw',
                   '/NAO/LElbowRoll', '/NAO/LWristYaw', '/NAO/LThumbBase', '/NAO/LThumbBase/joint',
                   '/NAO/LRFingerBase', '/NAO/LRFingerBase/joint', '/NAO/LRFingerBase/joint/Cuboid/joint',
                   '/NAO/LLFingerBase', '/NAO/LLFingerBase/joint', '/NAO/LLFingerBase/joint/Cuboid/joint',
                   '/NAO/HeadYaw', '/NAO/HeadPitch']

    clientID = connect_to_simulation()

    # Start the simulation
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    
    # Get and print joint states
    state = get_joint_states(clientID, joint_names)
    print("Joint States:", state)
    
    # Stop the simulation
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    
    # Close the connection
    sim.simxFinish(clientID)
