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
                             '/NAO/RRFingerBase/joint/Cuboid', '/NAO/RRFingerBase/joint/Cuboid/joint/Cuboid', '/NAO/RRFingerBase/joint/Cuboid/joint/Cuboid/part_',
                             '/NAO/RRFingerBase/joint/part_', '/NAO/RRFingerBase/part_', '/NAO/RLFingerBase/Cuboid',
                             '/NAO/RLFingerBase/joint/Cuboid', '/NAO/RLFingerBase/joint/Cuboid/joint/Cuboid', '/NAO/RLFingerBase/joint/Cuboid/joint/Cuboid/part_',
                             '/NAO/RLFingerBase/joint/part_', '/NAO/RLFingerBase/part_', '/NAO/part_36_sub',
                             '/NAO/RElbowYaw/part_', '/NAO/RShoulderPitch/part_', '/NAO/LShoulderPitch/shoulder_pitch_respondable',
                             '/NAO/LShoulderRoll/shoulder_roll_link_respondable',
                             '/NAO/LElbowYaw/elbow_yaw_link_respondable', '/NAO/LElbowRoll/elbow_roll_link_respondable',
                             '/NAO/LWristYaw/wrist_yaw_link_respondable', '/NAO/LThumbBase/Cuboid', '/NAO/LThumbBase/joint/Cuboid',
                             '/NAO/LThumbBase/joint/part_', '/NAO/LThumbBase/part_', '/NAO/LRFingerBase/Cuboid',
                             '/NAO/LRFingerBase/joint/Cuboid', '/NAO/LRFingerBase/joint/Cuboid/joint/Cuboid', '/NAO/LRFingerBase/joint/Cuboid/joint/Cuboid/part_',
                             '/NAO/LRFingerBase/joint/part_', '/NAO/LRFingerBase/part_', '/NAO/LLFingerBase/Cuboid',
                             '/NAO/LLFingerBase/joint/Cuboid', '/NAO/LLFingerBase/joint/Cuboid/joint/Cuboid', '/NAO/LLFingerBase/joint/Cuboid/joint/Cuboid/part_',
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
        # Base reward for forward movement 
        reward = state[0]  # Assuming state[0] indicates forward movement
        print(f"Forward movement reward: {reward}")

        # Check if the robot has fallen using the check_done method
        if self.check_done(state):
            reward -= 100  # Apply a large penalty if the robot has fallen
            print("Robot has fallen. Applying fall penalty: -100")
            return reward

        # Adding small bonus for staying upright
        upright_bonus = 0.1
        reward += upright_bonus
        print(f"Total calculated reward: {reward}")

        return reward

    
    def check_done(self, state):
        # Get the robot's head handle
        res_head, head_handle = sim.simxGetObjectHandle(self.clientID, '/NAO/HeadPitch', sim.simx_opmode_blocking)

        if res_head == sim.simx_return_ok:
            # Check the head's z-coordinate to determine if it has fallen
            _, head_position = sim.simxGetObjectPosition(self.clientID, head_handle, -1, sim.simx_opmode_blocking)
        
            # Define a z-coordinate threshold below which the head should not go
            head_z_threshold = 0.4  # Adjust this threshold according to the robot's height
        
            if head_position[2] < head_z_threshold:
                print(f"Head has fallen down (z position low): {head_position[2]}")
                return True

        return False  # The robot has not fallen


    
    def check_feet_touching_ground(self):
        # Get foot handles
        left_foot_handles = ['NAO/LFsrRR/sole_link_pure', 'NAO/LFsrFR/sole_link_pure', 
                            'NAO/LFsrRL/sole_link_pure', 'NAO/LFsrFL/sole_link_pure']
        right_foot_handles = ['NAO/RFsrRR/sole_link_pure', 'NAO/RFsrFR/sole_link_pure', 
                            'NAO/RFsrRL/sole_link_pure', 'NAO/RFsrFL/sole_link_pure']

        all_foot_handles = left_foot_handles + right_foot_handles

        for foot in all_foot_handles:
            res, foot_handle = sim.simxGetObjectHandle(self.clientID, foot, sim.simx_opmode_blocking)
            _, foot_position = sim.simxGetObjectPosition(self.clientID, foot_handle, -1, sim.simx_opmode_blocking)

            # Define a ground contact threshold
            ground_contact_threshold = 0.05
            if foot_position[2] > ground_contact_threshold:
                print(f"Foot {foot} is off the ground, robot might have fallen.")
                return True  # At least one foot is off the ground
        return False

    def check_head_position(self):
        # Get head handle
        res, head_handle = sim.simxGetObjectHandle(self.clientID, '/NAO//HeadPitch/link_respondable', sim.simx_opmode_blocking)
    
        # Get the z position (height) of the head
        _, head_position = sim.simxGetObjectPosition(self.clientID, head_handle, -1, sim.simx_opmode_blocking)
    
        # Define threshold for detecting a fall
        head_height_threshold = 0.2  
    
        if head_position[2] < head_height_threshold:
            print("Head is too low, robot likely fallen.")
            return True  # Robot has fallen
        return False  # Robot is still standing

    
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

        time.sleep(1.5)  # Delay to make sure changes take effect

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



    
