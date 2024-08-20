import sim
import numpy as np
import time

class NaoEnvironment:
    def __init__(self, clientID):
        # Init env & load robot 
        self.clientID = clientID
        self.joint_names = ['/NAO/RHipYawPitch', '/NAO/RHipRoll', '/NAO/RHipPitch',
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
        self.joint_handles = self.get_joint_handles()
        self.state_size = len(self.joint_names) * 2 # For both position & velocity of each joint
        self.action_size = len(self.joint_names) # 1 action per joint 

    def get_joint_handles(self):
        # Get all joint handles once and store them
        joint_handles = {}
        for joint in self.joint_names:
            res, handle = sim.simxGetObjectHandle(self.clientID, joint, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                joint_handles[joint] = handle
        return joint_handles

    def calculate_reward(self, state):
        # Calc and return reward based on the state
        reward = state[0] # Placeholder for forward movement
        return reward
    
    def check_done(self, state):
        # Check if ep is done
        return False # Placeholder
    
    def step(self, action):
        # Apply action, get new state, calc reward, check if done
        for i, joint in enumerate(self.joint_names):
            handle = self.joint_handles[joint]  # Use pre-stored handle
            # Apply action to each joint
            sim.simxSetJointTargetPosition(self.clientID, handle, action[i], sim.simx_opmode_oneshot)
        # Advance simulation
        sim.simxSynchronousTrigger(self.clientID)
        next_state = self.get_state() 
        reward = self.calculate_reward(next_state)
        done = self.check_done(next_state)
        return next_state, reward, done

    def reset(self):
        # Rather than stopping and restarting the simulation, I will reset the robot's position and joints
        sim.simxPauseSimulation(self.clientID, sim.simx_opmode_blocking)  # Pause simulation instead of stopping

        # Resetting joint angles to the initial position
        for joint in self.joint_names:
            res, handle = sim.simxGetObjectHandle(self.clientID, joint, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                sim.simxSetJointTargetPosition(self.clientID, handle, 0.0, sim.simx_opmode_blocking)  # Reset to 0 angle
    
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)  # Resume simulation after resetting the robot
        state = self.get_state()  # Get the initial state after reset
        return state


    def get_state(self):
        # Gets current state of env
        state = []
        for joint in self.joint_names:
            res, handle = sim.simxGetObjectHandle(self.clientID, joint, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                # Get joint angle
                res, position = sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_blocking)
                if res == sim.simx_return_ok:
                    state.append(position)
                else:
                    print(f"Failed to get joint position for {joint}")
                # Get joint velocity
                res, velocity = sim.simxGetObjectFloatParameter(self.clientID, handle, 2012, sim.simx_opmode_blocking)
                if res == sim.simx_return_ok:
                    state.append(velocity)
                else:
                    print(f"Failed to get joint velocity for {joint}")
            else:
                print(f"Failed to get joint handle for {joint}")
        return np.array(state)


    
