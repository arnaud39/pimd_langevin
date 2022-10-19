import numpy as np
import matplotlib.pyplot as plt


save_positions = []

iterations = 1000
timestep = 0.005
V_0 = 1.0
a = 0.04


def potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    return V_0 * ((x / a) ** 2 - 1) ** 2


def d_potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    return 4 * V_0 * x * ((x / a) ** 2 - 1)


def verlet(inital_positions, initial_velocities, force, iterations, timestep, mass):
      
  positions = [inital_positions]
  velocities = [initial_velocities]
  for i in range(iterations):
        new_position = positions[-1] + velocities[-1]*timestep + (force(positions[-1])/ (2*mass) )*(timestep**2)
        new_velocity = velocities[-1] + timestep*(force(new_position) + force(positions[-1]))

        positions.append(new_position)
        velocities.append(new_velocity)
  return positions, velocities


positions, velocities = verlet(
  inital_positions=a + 0.001, 
  initial_velocities=0 , 
  force=d_potential, 
  iterations=1000, 
  timestep=timestep, 
  mass=1.67e-27)
  
plt.plot(positions)
plt.show()