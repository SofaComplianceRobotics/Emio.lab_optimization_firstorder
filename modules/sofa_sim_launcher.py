import copy

import numpy as np

DEBUG = False

def createScene(rootnode):
    createScene2(rootnode, 3.5e4)

def createScene2(rootnode, youngModulus):
    from parts.emio import Emio
    from utils.header import addHeader, addSolvers
    from parts.controllers.assemblycontroller import AssemblyController
    from math import pi

    settings, modelling, simulation = addHeader(rootnode, inverse=False)

    rootnode.dt = 0.01
    rootnode.gravity = [0., -9810., 0.]
    addSolvers(simulation)
    rootnode.VisualStyle.displayFlags.value = ["hideBehavior"]

    emio = Emio(name="Emio",
                legsName=["blueleg", "blueleg", "blueleg", "blueleg"],
                legsModel=["beam"],
                legsPositionOnMotor=["counterclockwisedown","clockwisedown", "counterclockwisedown", "clockwisedown"],
                legsYoungModulus=[youngModulus],
                centerPartName="bluepart",
                centerPartType="rigid",
                extended=True)
    if not emio.isValid():
        return
    
    for motor in emio.motors:
        motor.addObject("JointConstraint", name="JointActuator", 
                        minDisplacement=-pi, maxDisplacement=pi,
                        index=0, value=0, valueType="displacement")

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    assembly = AssemblyController(emio)
    assembly.duration = 0.2
    emio.addObject(assembly)

    # Add effector
    emio.effector.addObject("MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1])
    emio.effector.addObject("RigidMapping", index=0)

    return rootnode


def run_forward_simulation(young_modulus, motor_angles: list[list[float]]) -> np.ndarray:
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
    createScene2(root, young_modulus)

    Sofa.Simulation.init(root)

    # Set motor angles
    emio = root.Simulation.Emio

    dt = 0.01

    # Run for assembly
    for _ in range(25):
        Sofa.Simulation.animate(root, dt)

    # Run the simulation for a fixed number of steps
    positions = []
    for angles in motor_angles:
        for i in range(4):
            emio.getChild(f"Motor{i}").JointActuator.value = angles[i]

        last_position = None
        position_change = None

        for _ in range(25):
            Sofa.Simulation.animate(root, dt)
            p = emio.CenterPart.Effector.getMechanicalState().position[0]
            if DEBUG:
                print("Current position:", p)
            if last_position is None:
                last_position = copy.deepcopy(p)
                continue
            else:
                position_change = np.linalg.norm(np.array(p) - np.array(last_position))
                last_position = copy.deepcopy(p)
                if DEBUG:
                    print(position_change)
        if position_change > 1e-1:
            print("Warning: Simulation may not have converged. Final position change:", position_change)

        # Retrieve the position of the effector
        positions.append(copy.deepcopy(emio.CenterPart.Effector.getMechanicalState().position[0][:3])) # Return as numpy array (first 3 components: x, y, z)

    return np.array(positions)


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
