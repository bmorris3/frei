import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_p, G, h, c, sigma_sb
from expecto import get_spectrum

from .twostream import flux_unit

__all__ = [
    'emission_to_teff',
    'get_binned_phoenix_spectrum'
]

def emission_to_teff(F_2_up):
    bbtemp = ((F_2_up.to(u.erg/u.s/u.cm**2, u.spectral_density(lam)) / sigma_sb)**(1/4)).decompose()
    # print(bbtemp, bbtemp.max())
    plt.semilogx(lam, bbtemp)

    retrieve_phoenix_temperature = np.average(bbtemp, weights=F_2_up.value) + 250 * u.K

    # retrieve_phoenix_temperature = np.percentile(bbtemp, 95)
    return retrieve_phoenix_temperature


def resolution(group):
    # https://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html
    wl = group.wavelength
    op = group.values
    
    wl_values = wl.values
    Delta_x = wl_values.max() - wl_values.min()
    result = np.trapz(op, wl.values) / Delta_x
    return xr.DataArray(data=result, 
                        dims=group.dims, 
                        coords=dict(
                            wavelength=[wl_values.mean()],
                        ), name='opacity')

def get_binned_phoenix_spectrum(temperature, g):
    spec = get_spectrum(temperature.value, log_g=np.log10(g.cgs.value), cache=True)
    phoenix_xr = xr.DataArray(
        spec.flux.to(flux_unit).value, dims=['wavelength'],
        coords=dict(wavelength=spec.wavelength.to(u.um).value)
    )
    phoenix_groups = phoenix_xr.groupby_bins("wavelength", wl_bins)
    phoenix_lowres = phoenix_groups.map(resolution)
    phoenix_lowres_padded = np.pad(phoenix_lowres, (0, len(lam) - phoenix_lowres.shape[0]))