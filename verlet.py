import numpy as np
import matplotlib.pyplot as plt
from typing import function
from scipy.constants import hbar, k
from typing import Tuple

save_positions = []

iterations = 1000
timestep = 0.005
V_0 = 1.0
a = 0.04

Parameters = {
    "energy": 7.24e-21, #V_0
    "distance": 4e-1, #Amstrom
    "mass": 1.67e-27, #mass proton
    "time": 10e-14, #computed to have hbar big enough
}

k_B = k / Parameters["energy"]  # 1.9e-3
hbar = hbar / Parameters["energy"] / Parameters["time"]  # 1.4e-1

P = 3


def potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    """Given potential for the problem."""
    return V_0 * ((x / a) ** 2 - 1) ** 2


def force_potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    """Gradient of the given potential."""
    return - 4 * V_0 * x * ((x / a) ** 2 - 1) / (a**2 * len(x))


def energy(
    x: np.array, v: np.array, m: float = 1.0, V_0: float = 1.0, a: float = 1.0
) -> tuple(np.array, np.array):
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * (v) ** 2 / 2
    return potential(x, V_0, a), kinetic_e


def K(M: float = 1.0, mass: float = 1.0, T: float = 1.0) -> float:
    return P * mass * (k_B * T / hbar) ** 2


def random_force(
    v: np.array,
    mass: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
    generated_number: float = np.random.random(len(v)),
) -> np.array:
    """Generate the Langegin random force."""
    return np.sqrt(2 * m * gamma * k_B * T) * generated_number - m * gamma * v


def force_PI(x: np.array, mass: float, T: float) -> np.array:
    """Path integral force."""
    return -K(mass, T) * (2 * x - np.roll(x, 1) - np.roll(x, -1))


def force(
    x: np.array, 
    v:np.array, 
    mass: float, 
    T: float, 
    k_B: float = 1.0, 
    gamma: float = 1.0, 
    V_0: float = 1.0, 
    a: float = 1.0
    ) -> np.array:

    force_total = force_potential(x, V_0, a) + force_PI(x, mass, T) + random_force(x, v, mass, gamma, k_B, T)
    return force_total


def verlet(
    inital_positions: np.array, 
    initial_velocities: np.array, 
    force: function, 
    iterations: int, 
    timestep: float, 
    mass: float, 
    T: float,
    k_B: float,
    gamma: float,
    V_0: float,
    a: float,
    ) -> tuple(list(np.array), list(np.array)):

    positions = [inital_positions]
    velocities = [initial_velocities]

    X_1 = positions[0] + timestep * velocities[0]  + timestep**2 * force(positions[0], velocities[0], mass, T, k_B, gamma, V_0, a) / (2*mass)
    V_1 = velocities[0] + timestep * force(positions[0], velocities[0],*args)

    X_2 = positions[1] + timestep * velocities[1]  + timestep**2 * force(positions[1], velocities[1], args) / (2*mass)
    V_2 = velocities[1] + timestep * force(positions[1], velocities[1],*args)

    for k in range(iterations):
          new_position, new_velocity = _verlet_step(positions, velocities, force, timestep, mass, T)
          positions.append(new_position)
          velocities.append(new_velocity)

    return positions, velocities


def _verlet_step(
    positions: list(np.array), 
    velocities: list(np.array), 
    force: function, 
    timestep: float,
    mass: float, 
    T:float
    ) -> tuple(np.array, np.array):
    new_position = 2 * positions[-1] - positions[-2] + timestep**2 * force(positions[-1], velocities[-1]) / (2*mass)  
    new_velocity = (3 * positions[-1] - 4 * positions[-2] + positions[-3]) / (2 * timestep)
    return new_position, new_velocity


positions, velocities = verlet(
  inital_positions=a + 0.001, 
  initial_velocities=0 , 
  force=d_potential, 
  iterations=1000, 
  timestep=timestep, 
  mass=1.67e-27)
  
plt.plot(positions)
plt.show()
    