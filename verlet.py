import numpy as np
import matplotlib.pyplot as plt
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


def potential(x: np.array, V_0: float = 1.0, a: float = 1.0):
    """Given potential for the problem."""
    return V_0 * ((x / a) ** 2 - 1) ** 2


def force_potential(x: np.array, V_0: float = 1.0, a: float = 1.0):
    """Gradient of the given potential."""
    return - 4 * V_0 * x * ((x / a) ** 2 - 1) / (a**2 * len(x))


def energy(x: np.array, v: np.array, m: float = 1.0, V_0: float = 1.0, a: float = 1.0):
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * (v) ** 2 / 2
    return potential(x, V_0, a), kinetic_e


def K(M: float = 1.0, mass: float = 1.0, T: float = 1.0):
    k = P * mass * (k_B * T / hbar) ** 2
    return k


def random_force(
    v: np.array,
    mass: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
    ):
    """Generate the Langegin random force."""
    generated_number = np.random.random(len(v))
    random_f = np.sqrt(2 * mass * gamma * k_B * T) * generated_number - mass * gamma * v

    return random_f


def force_PI(x: np.array, mass: float, T: float):
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
    ):
    force_total = force_potential(x, V_0, a) + force_PI(x, mass, T) + random_force(v, mass, gamma, k_B, T)
    return force_total


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

  
plt.plot(np.array(positions).transpose()[0])
plt.show()
