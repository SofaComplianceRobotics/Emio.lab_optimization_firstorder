import numpy as np

DEBUG = True

def createScene(rootnode, youngModulus):
    from parts.emio import Emio
    from utils.header import addHeader, addSolvers
    from parts.controllers.assemblycontroller import AssemblyController
    from math import pi

    settings, modelling, simulation = addHeader(rootnode, inverse=False)

    rootnode.dt = 0.01
    rootnode.gravity = [0., -9810., 0.]
    addSolvers(simulation)
    rootnode.VisualStyle.displayFlags.value = ["hideBehavior"]

    extended = True
    emio = Emio(name="Emio",
                legsName=["blueleg", "blueleg", "blueleg", "blueleg"],
                legsModel=["beam"],
                legsPositionOnMotor=["counterclockwisedown","clockwisedown", "counterclockwisedown", "clockwisedown"],
                legsYoungModulus=[youngModulus],
                centerPartName="bluepart",
                centerPartType="rigid",
                extended=extended)
    if not emio.isValid():
        return
    
    for motor in emio.motors:
        motor.addObject("JointConstraint", name="JointActuator", 
                        minDisplacement=-pi, maxDisplacement=pi,
                        index=0, value=0, valueType="displacement")

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    emio.addObject(AssemblyController(emio))

    # Add effector
    emio.effector.addObject("MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1])
    emio.effector.addObject("RigidMapping", index=0)

    print("Legs Young modulus: ", emio.Leg0.Leg0RigidBase.RigidifiedPoints.Leg.BeamInterpolation.defaultYoungModulus.value)

    return rootnode


def run_forward_simulation(young_modulus, motor_angles):
    """
    Simulate forward kinematics using SOFA simulation.

    Args:
        young_modulus: Young's modulus value for the material
        motor_angles: Array of motor angles

    Returns:
        End-effector position as numpy array [x, y, z]
    """
    import parameters
    import Sofa
    import SofaRuntime

    # Create the Sofa simulation scene
    SofaRuntime.importPlugin("Sofa.Component")
    SofaRuntime.importPlugin("Sofa.GUI.Component")
    SofaRuntime.importPlugin("Sofa.GL.Component")
    root = Sofa.Core.Node("root")
    # Set the Young's modulus parameter
    parameters.youngModulus = young_modulus

    # important to call this after parameters has been set.
    createScene(root, young_modulus)

    Sofa.Simulation.init(root)

    # Set motor angles
    emio = root.Simulation.Emio

    for i in range(4):
        emio.getChild(f"Motor{i}").JointActuator.value = motor_angles[i]


    # Run the simulation for a fixed number of steps
    dt = 0.01
    for _ in range(50):
        Sofa.Simulation.animate(root, dt)
        if DEBUG:
            position = emio.CenterPart.Effector.getMechanicalState().position[0]

    # Retrieve the position of the effector
    position = emio.CenterPart.Effector.getMechanicalState().position[0]
    print(position[:3].round(2))

    # Return as numpy array (first 3 components: x, y, z)
    return np.array(position[:3])


def main():
    """
    Main function for testing the forward simulation.
    """
    myYoungModulus = 2800
    myMotorAngles = [0.0, 0.0, 0.0, 0.0]  # Default motor angles

    position = run_forward_simulation(myYoungModulus, myMotorAngles)
    print(f"End-effector position: {position}")


if __name__ == "__main__":
    main()
