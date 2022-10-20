from scipy.constants import hbar, k
from typing import Tuple

import numpy as np


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