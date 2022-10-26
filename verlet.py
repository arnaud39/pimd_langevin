import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, k
from typing import Tuple
from dynamics import force, Parameters

mass = Parameters["mass"]


def verlet(
    inital_positions: np.array,
    initial_velocities: np.array,
    iterations: int,
    timestep: float,
):

    positions = [inital_positions]
    velocities = [initial_velocities]

    X_1 = (
        positions[0]
        + timestep * velocities[0]
        + timestep**2 * force(x=positions[0], v=velocities[0]) / (2 * mass)
    )
    V_1 = velocities[0] + timestep * force(x=positions[0], v=velocities[0])
    positions.append(X_1)
    velocities.append(V_1)

    X_2 = (
        positions[1]
        + timestep * velocities[1]
        + timestep**2 * force(x=positions[1], v=velocities[1]) / (2 * mass)
    )
    V_2 = velocities[1] + timestep * force(x=positions[1], v=velocities[1])
    positions.append(X_2)
    velocities.append(V_2)

    for k in range(iterations - 3):
        new_position, new_velocity = _verlet_step(
            positions, velocities, timestep=timestep
        )
        positions.append(new_position)
        velocities.append(new_velocity)

    return positions, velocities


def _verlet_step(
    positions: np.array,
    velocities: np.array,
    timestep: float,
):
    new_position = (
        2 * positions[-1]
        - positions[-2]
        + timestep**2 * force(x=positions[-1], v=velocities[-1]) / (2 * mass)
    )
    new_velocity = (3 * positions[-1] - 4 * positions[-2] + positions[-3]) / (
        2 * timestep
    )
    return new_position, new_velocity


# def V_0(C, a, mass):
#     return hbar**2 / (2 * mass * a**2 * C)
