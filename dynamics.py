from scipy.constants import hbar, k
from typing import Tuple

import numpy as np


Parameters = {
    "energy": 7.24e-21,  # V_0
    "distance": 4e-1,  # Amstrom
    "mass": 1.67e-27,  # mass proton
    "time": 10e-14,  # computed to have hbar big enough
}

T = 0.4 * 7.24e-21 / k
k_B = k / Parameters["energy"]  # 1.9e-3
hbar = hbar / Parameters["energy"] / Parameters["time"]  # 1.4e-1

P = 3


def potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    """Given potential for the problem."""
    return V_0 * ((x / a) ** 2 - 1) ** 2


def d_potential(x: np.array, V_0: float = 1.0, a: float = 1.0) -> np.array:
    """Gradient of the given potential."""
    return (4 * V_0 * x * ((x / a) ** 2 - 1)) / (-len(x) * a**2)


def energy(
    x: np.array, v: np.array, m: float = 1.0, V_0: float = 1.0, a: float = 1.0
) -> Tuple[np.array, np.array]:
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * (v) ** 2 / 2
    return potential(x, V_0, a), kinetic_e


def K(x, m: float = 1.0, T: float = 1.0) -> float:
    return len(x) * m * (k_B * T / hbar) ** 2


def random_force(
    v: np.array,
    m: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
) -> np.array:
    """Generate the Langegin random force."""
    generated_number = np.random.random(len(v))
    return np.sqrt(2 * m * gamma * k_B * T) * generated_number - m * gamma * v


def force_PI(x: np.array, m: float, T: float) -> np.array:
    """Path integral force."""
    return -K(x=x, m=m, T=T) * (2 * x - np.roll(x, 1) - np.roll(x, -1))


def force(
    x: np.array,
    v: np.array,
    m: float = 1.0,
    gamma: float = 1.0,
    V_0: float = 1.0,
    a: float = 1.0,
    T: float = T,
    k_B: float = k_B,
):
    return (
        random_force(v=v, m=m, gamma=gamma, k_B=k_B, T=T)
        + force_PI(x=x, m=m, T=T)
        + d_potential(x=x, V_0=V_0, a=a)
    )
