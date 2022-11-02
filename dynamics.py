from scipy.constants import hbar, k
from typing import Tuple

import numpy as np


Parameters = {
    "energy": 7.24e-21,  # V_0
    "distance": 4e-10,  # Amstrom
    "mass": 1.67e-27,  # mass proton
    "time": 10e-14,  # computed to have hbar big enough
}

T_0 = 0.4 * 7.24e-21 / k
k_B = k / Parameters["energy"]  # 1.9e-3
hbar = hbar / Parameters["energy"] / Parameters["time"]  # 1.4e-1


def potential(x: np.array, V_0: float = 1, a: float = 1.0) -> np.array:
    """Given potential for the problem."""
    return V_0 * ((x / a) ** 2 - 1) ** 2


def force_potential(x: np.array, V_0: float = 0.3, a: float = 1.0) -> np.array:
    """Gradient of the given potential."""
    return -(4 * V_0 * x * ((x / a) ** 2 - 1)) / (len(x) * a**2)


def energy(
    x: np.array, v: np.array, m: float = 1.0, V_0: float = 1.0, a: float = 1.0
) -> Tuple[np.array, np.array]:
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * v ** 2 / 2
    potential_e = potential(x, V_0, a)
    return potential_e, kinetic_e


def K(x, m: float = 1.0, T: float = 1.0) -> float:
    return len(x) * m * (k_B * T / hbar) ** 2


def random_force(
    v: np.array,
    timestep: float,
    m: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
) -> np.array:
    """Generate the Langegin random force."""
    generated_number = np.random.normal(size=len(v))
    return (
        np.sqrt(2 * m * gamma * k_B * T /  timestep) * generated_number - m * gamma * v
    )


def force_PI(x: np.array, m: float, T: float) -> np.array:
    """Path integral force."""
    return -K(x=x, m=m, T=T) * (2 * x - np.roll(x, 1) - np.roll(x, -1))


def force(
    x: np.array,
    v: np.array,
    timestep: float,
    m: float = 1.0,
    gamma: float = 1.0,
    V_0: float = 1,
    a: float = 1.0,
    T: float = T_0,
    k_B: float = k_B,
    termostat: bool = True,
):
    gamma = gamma * Parameters["time"]
    if termostat:
        force = random_force(v=v, timestep=timestep, m=m, gamma=gamma, k_B=k_B, T=T) + force_PI(x=x, m=m, T=T) + force_potential(x=x, V_0=V_0, a=a)
    else:
        force = force_PI(x=x, m=m, T=T) + force_potential(x=x, V_0=V_0, a=a)
    return force

