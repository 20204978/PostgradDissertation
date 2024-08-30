import qi
import argparse
import sys
import sim
import time

# Define your VREP connection function
def connect_to_vrep():
    sim.simxFinish(-1)  # Close any open connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID == -1:
        print("Failed to connect to VREP")
        sys.exit()
    print("Connected to VREP")
    return clientID

def main(session, clientID):
    # Set up NAO in VREP (CoppeliaSim) with basic movements
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    motion_service.wakeUp()
    posture_service.goToPosture("StandInit", 0.5)

    # Implement a basic walking pattern
    X = 0.5  # Forward velocity
    Y = 0.0  # Lateral velocity
    Theta = 0.0  # Rotation
    Frequency = 0.5  # Moderate speed
    try:
        motion_service.moveToward(X, Y, Theta, [["Frequency", Frequency]])
    except Exception as e:
        print(str(e))
        exit()

    time.sleep(4.0)
    motion_service.rest()

    # Close connection to VREP
    sim.simxFinish(clientID)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. Use '127.0.0.1' for VREP.")
    parser.add_argument("--port", type=int, default=9559,
                        help="NAOqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print(f"Can't connect to Naoqi at ip {args.ip} on port {args.port}.")
        sys.exit(1)

    clientID = connect_to_vrep()
    main(session, clientID)
