import numpy as np
import astropy.units as u

__all__ = [
    'pressure_grid', 
    'temperature_grid'
]

def pressure_grid(
    n_layers = 30,
    P_toa = -6,  # log10(bar)
    P_boa = 1.1,  # log10(bar)
):
    """
    Produce a pressure grid from bottom to top of atmosphere.
    """
    pressures = np.logspace(P_toa, P_boa, n_layers)[::-1] * u.bar
    return pressures

def temperature_grid(
    pressures,
    T_ref = 2300 * u.K,
    P_ref = 0.1 * u.bar
):
    """
    Produce a rough initial temperature grid for each pressure.
    """
    temperatures = T_ref * (pressures / P_ref) ** 0.1  # K
    return temperatures