import numpy as np
import xarray as xr
import astropy.units as u
from expecto import get_spectrum

from .twostream import flux_unit

__all__ = [
    'get_binned_phoenix_spectrum'
]


def resolution(group):
    Delta_x = group.wavelength.max() - group.wavelength.min()
    return group.integrate('wavelength').expand_dims(
        dict(wavelength=[group.wavelength.mean()])
    ) / Delta_x


def get_binned_phoenix_spectrum(T_eff, g, wl_bins, lam, cache=True):
    """
    Return a binned PHOENIX spectrum with effective temperature T_eff.

    Parameters
    ----------
    T_eff : ~astropy.units.Quantity
        Effective temperature of the PHOENIX model to retrieve
    g : ~astropy.units.Quantity
        Surface gravity
    wl_bins : ~astropy.units.Quantity
        Wavelength bin edges
    lam : ~astropy.units.Quantity
        Wavelength bin centers
    cache : bool
        Cache downloaded PHOENIX spectrum

    Returns
    -------
    binned_spectrum : ~astropy.units.Quantity
        PHOENIX spectrum binned to wavelength grid
    """
    spec = get_spectrum(T_eff.value, log_g=np.log10(g.cgs.value), cache=cache)
    phoenix_xr = xr.DataArray(
        spec.flux.to(flux_unit).value, dims=['wavelength'],
        coords=dict(wavelength=spec.wavelength.to(u.um).value)
    )
    phoenix_groups = phoenix_xr.groupby_bins("wavelength", wl_bins)
    phoenix_lowres = phoenix_groups.map(resolution)
    phoenix_lowres_padded = np.pad(
        phoenix_lowres, (0, len(lam) - phoenix_lowres.shape[0])
    ) * flux_unit
    return phoenix_lowres_padded
