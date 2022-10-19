import numpy as np

from scipy.constants import hbar, k

k_B = k
P = 3


def potential(x: np.array, V_0: float = 1., a: float= 1.):
    """Given potential for the problem."""
    return V_0 * ((x / a) ** 2 - 1) ** 2


def d_potential(x: np.array, V_0: float=1., a: float=1.):
    """Gradient of the given potential."""
    return 4 * V_0 * x * ((x / a) ** 2 - 1)


def energy(x: np.array, v: np.array, m: float = 1.0, V_0: float = 1.0, a: float = 1.0):
    """Compute potential and kinestic energy of the system."""
    kinetic_e = m * (v) ** 2 / 2
    return potential(x, V_0, a), kinetic_e


def K(M: float = 1.0, T: float = 1.0):
    return P * m * (k_B * T / hbar) ** 2


def random_force(
    v: np.array,
    m: float = 1.0,
    gamma: float = 1.0,
    k_B: float = 1.0,
    T: float = 1.0,
    generated_number: float = np.random.random(P),
):
    """Generate the Langegin random force."""
    return np.sqrt(2 * m * gamma * k_B * T) * generated_number - m * gamma * v


def force_PI(x= np.array, M: float, T: float):
    """Path integral force."""
    return -K(M, T) * (2*x - np.roll(x,1) - np.roll(x, -1))
