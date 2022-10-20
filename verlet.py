import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, k
from typing import Tuple
from dynamics import *

save_positions = []

iterations = 1000
timestep = 0.005
V_0 = 1.0
a = 0.04


def verlet(
    inital_positions: np.array, 
    initial_velocities: np.array, 
    force, 
    iterations: int, 
    timestep: float, 
    mass: float, 
    T: float,
    k_B: float,
    gamma: float,
    V_0: float,
    a: float,
    ):

    positions = [inital_positions]
    velocities = [initial_velocities]

    X_1 = positions[0] + timestep * velocities[0]  + timestep**2 * force(positions[0], velocities[0], mass, T, k_B, gamma, V_0, a) / (2*mass)
    V_1 = velocities[0] + timestep * force(positions[0], velocities[0], mass, T, k_B, gamma, V_0, a)
    positions.append(X_1)
    velocities.append(V_1)
    X_2 = positions[1] + timestep * velocities[1]  + timestep**2 * force(positions[1], velocities[1], mass, T, k_B, gamma, V_0, a) / (2*mass)
    V_2 = velocities[1] + timestep * force(positions[1], velocities[1], mass, T, k_B, gamma, V_0, a)
    positions.append(X_2)
    velocities.append(V_2)
    for k in range(iterations):
          new_position, new_velocity = _verlet_step(positions, velocities, force, timestep, mass, T, k_B, gamma, V_0, a)
          positions.append(new_position)
          velocities.append(new_velocity)

    return positions, velocities


def _verlet_step(
    positions, 
    velocities, 
    force, 
    timestep: float,
    mass: float, 
    T:float,
    k_B,
    gamma,
    V_0,
    a,
    ):
    new_position = 2 * positions[-1] - positions[-2] + timestep**2 * force(positions[-1], velocities[-1], mass, T, k_B, gamma, V_0, a) / (2*mass)  
    new_velocity = (3 * positions[-1] - 4 * positions[-2] + positions[-3]) / (2 * timestep)
    return new_position, new_velocity


positions, velocities = verlet(
    inital_positions= np.linspace(a,a,10), 
    initial_velocities=np.linspace(0,0,10), 
    force=force, 
    iterations= 10000, 
    timestep= timestep,
    mass=1, 
    T=1,
    k_B=1,
    gamma= 1,
    V_0= 1,
    a=1,
    )

  
plt.plot(np.array(positions).transpose()[-1])
plt.show()
