from scipy.constants import hbar, k
from typing import Tuple

import numpy as np


Parameters = {
    "energy": 7.24e-21,  # V_0
    "distance": 4e-1,  # Amstrom
    "mass": 1.67e-27,  # mass proton
    "time": 10e-14,  # computed to have hbar big enough
}

k_B = k / Parameters["energy"]  # 1.9e-3
hbar = hbar / Parameters["energy"] / Parameters["time"]  # 1.4e-1

P = 3


def potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    """Given potential for the problem."""
    return V_0 * ((x / a) ** 2 - 1) ** 2


def d_potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    """Gradient of the given potential."""
    return 4 * V_0 * x * ((x / a) ** 2 - 1)


def energy(
    x: np.array, v: np.array, m: float = 1.0, V_0: float = 1.0, a: float = 1.0
) -> Tuple[np.array, np.array]:
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * (v) ** 2 / 2
    return potential(x, V_0, a), kinetic_e


def K(M: float = 1.0, m: float = 1.0, T: float = 1.0) -> float:
    return P * m * (k_B * T / hbar) ** 2


def random_force(
    v: np.array,
    m: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
    generated_number: float = np.random.random(P),
) -> np.array:
    """Generate the Langegin random force."""
    return np.sqrt(2 * m * gamma * k_B * T) * generated_number - m * gamma * v


def force_PI(x: np.array, M: float, T: float) -> np.array:
    """Path integral force."""
    return -K(M, T) * (2 * x - np.roll(x, 1) - np.roll(x, -1))
