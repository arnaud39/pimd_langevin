from scipy.constants import hbar, k
import numpy as np

k_B = k
P = 1


def potential(x, V_0: float, a: float):
    return V_0 * ((x / a) ** 2 - 1) ** 2


def d_potential(x, V_0: float, a: float):
    return 4 * V_0 * x * ((x / a) ** 2 - 1)


def energy(x, v, m: float = 1.0, V_0: float = 1.0, a: float = 1.0):
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * (v) ** 2 / 2
    return potential(x, V_0, a), kinetic_e


def K(M: float = 1.0, T: float = 1.0):
    return P * m * (k_B * T / hbar) ** 2


def random_force(
    v,
    m: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
    generated_number: float = np.random.random(P),
):
    return np.sqrt(2 * m * gamma * k_B * T) * generated_number - m * gamma * v


def force(x, M: float, T: float):
    return -2.0 * K(M, T)
