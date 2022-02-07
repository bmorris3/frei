from glob import glob
import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_p, G, h, c
import xarray as xr
import dask

from .chemistry import chemistry

__all__ = [
    'binned_opacity',
    'kappa'
]

interp_kwargs = dict(
    method='nearest', 
    kwargs=dict(fill_value="extrapolate")
)

def mapfunc_exact(group, temperature=2500, pressure=1e-08, interp_kwargs=interp_kwargs):
    # https://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html
    wl = group.wavelength
    op = group.interp(
        temperature=temperature, pressure=pressure, **interp_kwargs
    ).opacity
    
    Delta_x = wl.max() - wl.min()
    result = xr.apply_ufunc(
        np.trapz, op, wl, dask='allowed', #dask='parallelized',
        input_core_dims=[["temperature", "pressure", "wavelength"],
                         ["wavelength"]],
        output_core_dims=[["temperature", "pressure"]],
    ) / Delta_x
    return result.expand_dims(dict(wavelength=[wl.mean()]))

def delayed_map_exact_concat(grouped, temperatures, pressures, lam, client):
    results = []
    for i, (name, group) in enumerate(grouped):
        results.append(
            dask.delayed(mapfunc_exact)(
                group, temperature=temperatures.value,
                pressure=pressures.to(u.bar).value
            )
        )

    r = client.compute(results)
    results = client.gather(r)
    # Concatenate the results from each delayed task into a big dask array
    return xr.concat(
        results, dim='wavelength'
        # Also interpolate to span all grid wavelength grid points that weren't covered by 
        # the xarray groupings (currently xarray doesn't return empty bins)
    ).interp(
        dict(wavelength=lam.to(u.um).value),
        method='linear', kwargs=dict(fill_value='extrapolate')
    )


def binned_opacity(
    path, temperatures, pressures, wl_bins, lam, client
):
    """
    Compute opacity for all available species, binned to wavelengths lam.
    """
    if len(set(temperatures)) == 1: 
        # uniform temperature submitted, draw temperature grid from this temperature grid: 
        temperatures = np.linspace(500, 4000, len(pressures))[::-1] * u.K
        
    results = dict()    
    xr_kwargs = dict(chunks='auto')#, engine='h5netcdf')
    paths = glob(path)
    for i, path in enumerate(paths): 
        species_name = path.split('/')[-1].split('_')[0]
        
        species_ds = xr.open_dataset(path, **xr_kwargs)
        print(f'Loading opacity for {species_name} ' +
              f'({i+1}/{len(paths)}): group in wavelength')
        species_grouped = species_ds.groupby_bins("wavelength", wl_bins)
        print(f'Loading opacity for {species_name} ' +
              f'({i+1}/{len(paths)}): compute exact k-distributions')
        species_binned = delayed_map_exact_concat(
            species_grouped, temperatures, pressures, lam, client
        )
        results[species_name] = species_binned
        
        del species_ds, species_grouped
    return results


# Malik 2017 Eqn 17
n_lambda_H2 = lambda wavelength: 13.58e-5 * (
        1 + (7.52e-11 * u.cm**2) * wavelength**-2
) + 1
# Deitrick 2020 Eqn C3
n_lambda_He = lambda wavelength: 1e-8 * (
        2283 + (1.8102e13 / (1.5342e10 - (wavelength / (1 * u.um))**-2))
) + 1
n_ref_H2 = 2.68678e19 * u.cm**-3
n_ref_He = 2.546899e19 * u.cm**-3
K_lambda = 1
# Malik 2017 Eqn 16

def rayleigh_H2(wavelength, m_bar=2.4*m_p): 
    return ((24 * np.pi**3 / n_ref_H2**2 / wavelength**4 *
        ((n_lambda_H2(wavelength)**2 - 1) / 
         (n_lambda_H2(wavelength)**2 + 2))**2 * K_lambda
    ) / m_bar).decompose()

def rayleigh_He(wavelength, m_bar=2.4*m_p): 
    return ((24 * np.pi**3 / n_ref_He**2 / wavelength**4 *
        ((n_lambda_He(wavelength)**2 - 1) / 
         (n_lambda_He(wavelength)**2 + 2))**2 * K_lambda
    ) / m_bar).decompose()


def kappa(
    opacities,
    temperature, 
    pressure, 
    lam, 
    m_bar=2.4*m_p
): 
    """
    Return the opacity at a given temperature and pressure.
    """
    sigma_scattering = rayleigh_H2(lam, m_bar) + rayleigh_He(lam, m_bar)
    ops = [sigma_scattering]
    interp_kwargs = dict(
        method='linear', 
        kwargs=dict(fill_value='extrapolate')
    )
    
    fastchem_mmr = chemistry(
        u.Quantity([temperature]), u.Quantity([pressure]), m_bar=m_bar
    )
    
    for species in opacities: 
        ops.append(
            fastchem_mmr[species] * opacities[species].interp(
                dict(temperature=temperature.value,
                     pressure=pressure.to(u.bar).value),
            **interp_kwargs).values * u.cm**2 / u.g
        )

    return u.Quantity(ops).sum(axis=0), sigma_scattering
