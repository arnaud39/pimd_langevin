import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.constants import hbar, k
from typing import Tuple
from dynamics import force, Parameters, T_0

mass = 1  # Parameters["mass"]
time = Parameters["time"]


def verlet(
    inital_positions: np.array,
    initial_velocities: np.array,
    iterations: int,
    timestep: float,
    gamma: float,
    T: float = T_0,
    termostat: bool = True,
):
    timestep = timestep / time
    positions = [inital_positions]
    velocities = [initial_velocities]

    X_1 = (
        positions[0]
        + timestep * velocities[0]
        + timestep**2
        * force(x=positions[0], v=velocities[0], timestep=timestep, gamma=gamma, T=T, termostat=termostat)
        / mass / 2
    )
    V_1 = velocities[0] + timestep * force(
        x=positions[0], v=velocities[0], timestep=timestep, gamma=gamma, T=T, termostat=termostat
    ) / mass
    positions.append(X_1)
    velocities.append(V_1)

    X_2 = (
        positions[1]
        + timestep * velocities[1]
        + timestep**2
        * force(x=positions[1], v=velocities[1], timestep=timestep, gamma=gamma, T=T, termostat=termostat)
        / mass / 2
    )
    V_2 = velocities[1] + timestep * force(
        x=positions[1], v=velocities[1], timestep=timestep, gamma=gamma, T=T, termostat=termostat
    ) / mass
    positions.append(X_2)
    velocities.append(V_2)

    for k in tqdm(range(iterations - 3)):
        new_position, new_velocity = _verlet_step(
            positions, velocities, timestep=timestep, gamma=gamma, T=T, termostat=termostat
        )
        positions.append(new_position)
        velocities.append(new_velocity)

    return positions, velocities


def _verlet_step(
    positions: np.array,
    velocities: np.array,
    timestep: float,
    gamma: float,
    T: float,
    termostat: bool,
):
    new_position = (
        2 * positions[-1]
        - positions[-2]
        + timestep**2
        * force(x=positions[-1], v=velocities[-1], gamma=gamma, T=T, timestep=timestep, termostat=termostat)
        / mass
    )
    new_velocity = (3 * positions[-1] - 4 * positions[-2] + positions[-3]) / (
        2 * timestep
    )
    return new_position, new_velocity
