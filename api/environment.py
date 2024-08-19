import sim
import numpy as np

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
        self.state_size = len(self.joint_names) * 2 # For both position & velocity of each joint
        self.action_size = len(self.joint_names) # 1 action per joint 

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
            res, handle = sim.simxGetObjectHandle(self.clientID, joint, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                # Apply action to each joint
                sim.simxSetJointTargetPosition(self.clientID, handle, action[i], sim.simx_opmode_oneshot)
        # Advance simulation
        sim.simxSynchronousTrigger(self.clientID)
        next_state = self.get_state() 
        reward = self.calculate_reward(next_state)
        done = self.check_done(next_state)
        return next_state, reward, done

    def reset(self):
        # Reset env to initial state
        state = self.get_state
        return state

    def get_state(self):
        # Gets current state of env
        state = []
        for joint in self.joint_names:
            res, handle = sim.simxGetObjectHandle(self.clientID, joint, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                # Get joint angle
                res, position = sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_blocking)
                state.append(position)
                # Get join velocity
                res, velocity = sim.simxGetObjectFloatParameter(self.clientID, handle, 2012, sim.simx_opmode_blocking)
                state.append(velocity)
        return np.array(state)

    
