import sys
import time
from naoqi import ALProxy
import motion 

def StiffnessOn(proxy):
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

def main(robotIP):
    # Initialize proxies
    try:
        motionProxy = ALProxy("ALMotion", robotIP, 9559)
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except RuntimeError as e:
        print("Could not create proxy to ALMotion or ALRobotPosture")
        print("Error was: ", e)
        sys.exit(1)

    # Set NAO in Stiffness On
    StiffnessOn(motionProxy)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    #####################
    ## Enable arms control by Walk algorithm
    #####################
    motionProxy.setWalkArmsEnabled(True, True)

    #####################
    ## FOOT CONTACT PROTECTION
    #####################
    motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True]])

    #TARGET VELOCITY
    X = -0.5  #backward
    Y = 0.0
    Theta = 0.0
    Frequency =0.0 # low speed
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

    time.sleep(4.0)

    #TARGET VELOCITY
    X = 0.8
    Y = 0.0
    Theta = 0.0
    Frequency =1.0 # max speed
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

    time.sleep(4.0)

    #TARGET VELOCITY
    X = 0.2
    Y = -0.5
    Theta = 0.2
    Frequency =1.0
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

    time.sleep(2.0)

    #####################
    ## Arms User Motion
    #####################
    # Arms motion from user have always the priority than walk arms motion
    JointNames = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
    Arm1 = [-40,  25, 0, -40]
    Arm1 = [ x * motion.TO_RAD for x in Arm1]

    Arm2 = [-40,  50, 0, -80]
    Arm2 = [ x * motion.TO_RAD for x in Arm2]

    pFractionMaxSpeed = 0.6

    motionProxy.angleInterpolationWithSpeed(JointNames, Arm1, pFractionMaxSpeed)
    motionProxy.angleInterpolationWithSpeed(JointNames, Arm2, pFractionMaxSpeed)
    motionProxy.angleInterpolationWithSpeed(JointNames, Arm1, pFractionMaxSpeed)

    time.sleep(2.0)

    #####################
    ## End Walk
    #####################
    #TARGET VELOCITY
    X = 0.0
    Y = 0.0
    Theta = 0.0
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) > 1:
        robotIp = sys.argv[1]

    main(robotIp)
