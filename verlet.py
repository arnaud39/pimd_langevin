import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, k
from typing import Tuple
from dynamics import *
from dynamics_ad import *



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

    for k in range(iterations-3):
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
    k_B: float,
    gamma: float,
    V_0: float,
    a: float,
    ):
    new_position = 2 * positions[-1] - positions[-2] + timestep**2 * force(positions[-1], velocities[-1], mass, T, k_B, gamma, V_0, a) / (2*mass)  
    new_velocity = (3 * positions[-1] - 4 * positions[-2] + positions[-3]) / (2 * timestep)
    return new_position, new_velocity

def V_0(C, a, mass):
    return hbar**2 / (2 * mass * a**2 * C)

timestep = 10**(-17)
mass = 1.6726 * 10**(-27)
a = 4 * 10**(-11)
V_0 = V_0(0.3, a, mass)
k_B = 1.38*10**(-23)
T = 0.4 * V_0 / k_B 
initial_position = np.linspace(-10**(-11), -10**(-11),100)
initial_velocities = np.linspace(0,0,100)
force = force
gamma = 10**12
iterations = 1000000

positions, velocities = verlet(
    inital_positions=initial_position, 
    initial_velocities=initial_velocities, 
    force=force, 
    iterations= iterations, 
    timestep= timestep,
    mass=mass, 
    T=T,
    k_B= k_B,
    gamma= gamma,
    V_0= V_0,
    a=a,
    )

k = K(P=10, k_B=k_B, mass=mass, T=T)


print(hbar)
print(V_0)
print(T)
print(len(positions))
print(k)

fig, ax = plt.subplots(2, 1)

time = [timestep*i for i in range(iterations)]
ax[0].plot(time, np.array(positions).transpose()[-1])
ax[1].plot(time, np.array(velocities).transpose()[-1])
plt.show()
