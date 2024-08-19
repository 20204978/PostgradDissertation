import sim
import time

sim.simxFinish(-1) #closes any other connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if clientID != -1:
    print("Connected")
    res, joint_handle = sim.simxGetObjectHandle(clientID, '/NAO/RShoulderPitch', sim.simx_opmode_blocking)

    if res == sim.simx_return_ok:
        print("Joint handle obtained successfully")

        # Move the joint 45 degrees
        sim.simxSetJointTargetPosition(clientID, joint_handle, 0.5, sim.simx_opmode_oneshot)

        # Wait to see
        time.sleep(2)

        # Move back
        sim.simxSetJointTargetPosition(clientID, joint_handle, 0, sim.simx_opmode_oneshot)

    else:
        print("Failed to obtain joint handle")

    # Close conn
    sim.simxFinish(clientID)
else:
    print("Failed to connect")