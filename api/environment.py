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
        self.object_names = ['/NAO/hip_roll_link_respondable', '/NAO/RHipRoll/hip_roll_link_respondable', 
                             '/NAO/hip_pitch_pure', '/NAO/knee_pitch_link_pure', '/NAO/ankle_pitch_link_pure',
                             '/NAO/sole_link_pure', '/NAO/RAnkleRoll/part_', '/NAO/RFsrRR', 
                             '/NAO/RFsrRR/sole_link_pure', '/NAO/RFsrFR', '/NAO/RFsrFR/sole_link_pure',
                             '/NAO/RFsrRL', '/NAO/RFsrRL/sole_link_pure', '/NAO/RFsrFL',
                             '/NAO/RFsrFL/sole_link_pure', '/NAO/RHipPitch/part_', '/NAO/part_22_sub',
                             '/NAO/part_10_sub', '/NAO/RHipRoll/part_', '/NAO/part_',
                             '/NAO/LHipYawPitch/hip_roll_link_respondable', '/NAO/LHipRoll/hip_roll_link_respondable',
                             '/NAO/LHipYawPitch/hip_pitch_pure', '/NAO/LHipYawPitch/knee_pitch_link_pure',
                             '/NAO/LHipYawPitch/ankle_pitch_link_pure', '/NAO/ankle_link_pure', '/NAO/LAnkleRoll/part_',
                             '/NAO/LFsrRR', '/NAO/LHipYawPitch/sole_link_pure', '/NAO/LFsrFR',
                             '/NAO/LFsrFR/sole_link_pure', '/NAO/LFsrRL', '/NAO/LFsrRL/sole_link_pure',
                             '/NAO/LFsrFL', '/NAO/LFsrFL/sole_link_pure', '/NAO/LHipPitch/part_',
                             '/NAO/part_19_sub', '/NAO/part_13_sub', '/NAO/LHipRoll/part_',
                             '/NAO/LHipYawPitch/part_', '/NAO/shoulder_pitch_respondable', '/NAO/shoulder_roll_link_respondable',
                             '/NAO/elbow_yaw_link_respondable', '/NAO/elbow_roll_link_respondable',
                             '/NAO/wrist_yaw_link_respondable', '/NAO/RThumbBase/Cuboid', '/NAO/RThumbBase/joint/Cuboid',
                             '/NAO/RThumbBase/joint/part_', '/NAO/RThumbBase/part_', '/NAO/RRFingerBase/Cuboid',
                             '/NAO/RRFingerBase/joint/Cuboid', '/NAO/RRFingerBase/joint/Cuboid/joint/Cuboid', '/NAO/.../part_',
                             '/NAO/RRFingerBase/joint/part_', '/NAO/RRFingerBase/part_', '/NAO/RLFingerBase/Cuboid',
                             '/NAO/RLFingerBase/joint/Cuboid', '/NAO/RLFingerBase/joint/Cuboid/joint/Cuboid', '/NAO/.../part_',
                             '/NAO/RLFingerBase/joint/part_', '/NAO/RLFingerBase/part_', '/NAO/part_36_sub',
                             '/NAO/RElbowYaw/part_', '/NAO/RShoulderPitch/part_', '/NAO/.../shoulder_pitch_respondable',
                             '/NAO/.../shoulder_roll_link_respondable',
                             '/NAO/.../elbow_yaw_link_respondable', '/NAO/.../elbow_roll_link_respondable',
                             '/NAO/.../wrist_yaw_link_respondable', '/NAO/LThumbBase/Cuboid', '/NAO/LThumbBase/joint/Cuboid',
                             '/NAO/LThumbBase/joint/part_', '/NAO/LThumbBase/part_', '/NAO/LRFingerBase/Cuboid',
                             '/NAO/LRFingerBase/joint/Cuboid', '/NAO/.../Cuboid', '/NAO/.../part_',
                             '/NAO/LRFingerBase/joint/part_', '/NAO/LRFingerBase/part_', '/NAO/LLFingerBase/Cuboid',
                             '/NAO/LLFingerBase/joint/Cuboid', '/NAO/.../Cuboid', '/NAO/.../part_',
                             '/NAO/LLFingerBase/joint/part_', '/NAO/LLFingerBase/part_', '/NAO/part_12_sub',
                             '/NAO/LElbowYaw/part_', '/NAO/LShoulderPitch/part_', '/NAO/link_respondable',
                             '/NAO/HeadPitch/link_respondable', '/NAO/part_16_sub', '/NAO/vision[0]',
                             '/NAO/vision[1]', '/NAO/HeadYaw/part_', '/NAO/part_20_sub',]  
        self.joint_handles = self.get_joint_handles()
        self.initial_states = self.get_initial_states()  # Capture initial states for all objects
        self.state_size = len(self.joint_names) * 2  # For both position & velocity of each joint
        self.action_size = len(self.joint_names)  # 1 action per joint

    def get_joint_handles(self):
        # Get handles for all joints and objects
        joint_handles = {}
        for joint in self.joint_names + self.object_names:
            res, handle = sim.simxGetObjectHandle(self.clientID, joint, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                joint_handles[joint] = handle
        return joint_handles

    def get_initial_states(self):
        # Capture initial positions and orientations for all joints and objects
        initial_states = {}
        for joint in self.joint_names + self.object_names:
            handle = self.joint_handles.get(joint)
            if handle:
                res, position = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
                res, orientation = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_blocking)
                initial_states[joint] = {'position': position, 'orientation': orientation}
        return initial_states

    def calculate_reward(self, state):
        # Assume state[0] represents forward position along the x-axis
        # Assume state[1] represents forward velocity
        forward_position = state[0]
        forward_velocity = state[1]
        # Define thresholds
        min_velocity_threshold = 0.1  # Minimum forward velocity to be rewarded
        backward_threshold = -0.1     # Penalty if moving backward
        # Calculate the reward
        if forward_velocity < min_velocity_threshold:
            reward = -5  # Penalise for low speed
        elif forward_velocity < backward_threshold:
            reward = -10  # Penalise for moving backward
        else:
            reward = forward_velocity + forward_position # Reward based on speed and position
        return reward

    
    def check_done(self, state):
        # Assume state[2] represents the z-position of the robot's center of mass
        z_position = state[2]
        # Define a threshold for falling (robot falls if it gets too close to the ground)
        fall_threshold = 0.2  # Example z-position threshold where the robot is considered fallen
        if z_position < fall_threshold:
            return True  # End the episode if the robot has fallen
        return False  # Continue if the robot is still standing
    
    def step(self, action):
        # Apply action, get new state, calc reward, check if done
        if isinstance(action, int):
            action = [action] * len(self.joint_names)  # Repeat the single action for each joint
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
        # Pause the sim
        sim.simxPauseSimulation(self.clientID, sim.simx_opmode_blocking)

        # Reset all joints and objects to their initial states
        for joint, state in self.initial_states.items():
            handle = self.joint_handles.get(joint)
            if handle:
                sim.simxSetObjectPosition(self.clientID, handle, -1, state['position'], sim.simx_opmode_blocking)
                sim.simxSetObjectOrientation(self.clientID, handle, -1, state['orientation'], sim.simx_opmode_blocking)

        time.sleep(1)  # Delay to make sure changes take effect

        # Resume the simulation after resetting the robot
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)
        # Get the initial state after reset
        state = self.get_state()
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



    
