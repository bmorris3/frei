import numpy as np
import astropy.units as u

__all__ = [
    'pressure_grid',
    'temperature_grid'
]


def pressure_grid(
    n_layers=30,
    P_toa=-6,
    P_boa=1.1,
):
    """
    Produce a pressure grid from bottom to top of atmosphere.

    Parameters
    ----------
    n_layers : int
        Number of layers in the atmosphere
    P_toa : float
        Pressure at the top of the atmosphere in log10(bar)
    P_boa : float
        Pressure at the bottom of the atmosphere in log10(bar)

    Returns
    -------
    pressures : ~astropy.units.Quantity
        Pressure grid
    """
    pressures = np.logspace(P_toa, P_boa, n_layers)[::-1] * u.bar
    return pressures


def temperature_grid(
    pressures,
    T_ref=2300 * u.K,
    P_ref=0.1 * u.bar,
    alpha=0.1
):
    """
    Produce a rough initial temperature grid for each pressure.

    Parameters
    ----------
    pressures : ~astropy.units.Quantity
        Pressure grid
    T_ref : ~astropy.units.Quantity
        Temperature at reference pressure
    P_ref : ~astropy.units.Quantity
        Reference pressure
    alpha : float
        Power law index for initial temperature-pressure profile

    Returns
    -------
    pressures : ~astropy.units.Quantity
        Pressure grid
    """
    temperatures = T_ref * (pressures / P_ref) ** alpha
    return temperatures
