import os
import tarfile
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
import astropy.units as u
from astropy.constants import m_p
import xarray as xr

from .chemistry import chemistry
from .interp import groupby_bins_agg


__all__ = [
    'binned_opacity',
    'kappa',
    'load_example_opacity',
    'download_molecule', 
    'download_atom'
]

n_ref_H2 = 2.68678e19 * u.cm**-3
n_ref_He = 2.546899e19 * u.cm**-3
K_lambda = 1

interp_kwargs = dict(
    method='nearest', 
    kwargs=dict(fill_value="extrapolate")
)


def mapfunc_exact(
        group, temperature=2500, pressure=1e-08, interp_kwargs=interp_kwargs
):
    wl = group.wavelength
    op = group.interp(
        dict(temperature=temperature, pressure=pressure), **interp_kwargs
    ).opacity
    
    Delta_x = wl.max() - wl.min()
    return op.integrate('wavelength').expand_dims(dict(wavelength=[wl.mean()])) / Delta_x


def delayed_map_exact_concat(grouped, temperatures, pressures, lam, client):

    r = client.submit(
        grouped.map, mapfunc_exact,
        temperature=temperatures.value,
        pressure=pressures.to(u.bar).value
    )

    r = client.compute(r)
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
    temperatures, pressures, wl_bins, lam, groupies=True,
        species=None, path=None
):
    """
    Compute opacity for all available species, binned to wavelengths lam.

    Parameters
    ----------
    path : str
        Path passed to ~glob.glob to find opacity netCDF files.
    temperatures : ~astropy.units.Quantity
        Temperature grid
    pressures : ~astropy.units.Quantity
        Pressure grid
    wl_bins : ~astropy.units.Quantity
        Wavelength bin edges
    lam : ~astropy.units.Quantity
        Wavelength bin centers
    path : str (optional)
        Path to opacity files.
    species : list or None (optional)
        List of species to load. If None, load all available.
    
    Returns
    -------
    op : dict
        Opacity tables for each species
    """
    from .chemistry import iso_to_species

    if path is None: 
        path = os.path.join(os.path.expanduser('~'), '.frei', '*.nc')

    paths = glob(path)

    if species is None:
        # Get all species names
        species = [
            iso_to_species(path.split('/')[-1].split('_')[0]) for path in paths
        ]

    fetch_paths = [
        path for path in paths 
        if iso_to_species(path.split('/')[-1].split('_')[0]) in species
    ]
    fetch_species = [
        iso_to_species(path.split('/')[-1].split('_')[0]) 
        for path in fetch_paths
    ]

    results = dict()    
    xr_kwargs = dict(
         # chunks='auto'
    )

    pbar = tqdm(zip(fetch_species, fetch_paths), total=len(fetch_paths))

    if groupies:
        for species_name, species_path in pbar:
            isotopologue = species_path.split('/')[-1].split('_')[0]

            pbar.set_description(f'Opacities for {isotopologue} (opening)')
            species_ds = xr.open_dataset(species_path, **xr_kwargs)
            pbar.set_description(f'Opacities for {isotopologue} (cropping)')
            dropped = species_ds.opacity.where(
                (species_ds.wavelength > wl_bins.min()) &
                (species_ds.wavelength < wl_bins.max()),
                drop=True
            )
            pbar.set_description(f'Opacities for {isotopologue} (integrating)')
            group_by = groupby_bins_agg(
                dropped, dropped.wavelength, wl_bins, func=np.trapz
            ) * (wl_bins[1:] - wl_bins[:-1]) * 1e-3
            pbar.set_description(f'Opacities for {isotopologue} (interp)')
            results[isotopologue] = group_by.interp(
                dict(
                    temperature=temperatures.value,
                    pressure=pressures.to(u.bar).value
                ), **interp_kwargs
            )
            del species_ds, dropped
        return results

    for species_name, species_path in pbar:
        isotopologue = species_path.split('/')[-1].split('_')[0]

        pbar.set_description(f'Opacities for {isotopologue} (opening)')
        species_ds = xr.open_dataset(species_path, **xr_kwargs)
        pbar.set_description(f'Opacities for {isotopologue} (binning)')
        species_grouped = species_ds.groupby_bins("wavelength", wl_bins)
        pbar.set_description(f'Opacities for {isotopologue} (integrating)')
        species_binned = species_grouped.map(
            mapfunc_exact,
            temperature=temperatures.value,
            pressure=pressures.to(u.bar).value
        )
        pbar.set_description(f'Opacities for {isotopologue} (interp)')
        results[isotopologue] = species_binned.interp(
            dict(wavelength=lam.to(u.um).value),
            method='linear', kwargs=dict(fill_value='extrapolate')
        )

        del species_ds, species_grouped, species_binned
    return results


def n_lambda_H2(wavelength):
    # Malik 2017 Eqn 17
    return 13.58e-5 * (
        1 + (7.52e-11 * u.cm**2) * wavelength**-2
    ) + 1


def n_lambda_He(wavelength):
    # Deitrick 2020 Eqn C3
    return 1e-8 * (2283 +
        (1.8102e13 / (1.5342e10 - (wavelength / (1 * u.um))**-2))
    ) + 1


def rayleigh_H2(wavelength, m_bar=2.4*m_p):
    # Malik 2017 Eqn 16
    return ((24 * np.pi**3 / n_ref_H2**2 / wavelength**4 *
        ((n_lambda_H2(wavelength)**2 - 1) / 
         (n_lambda_H2(wavelength)**2 + 2))**2 * K_lambda
    ) / m_bar).decompose()


def rayleigh_He(wavelength, m_bar=2.4*m_p):
    # Malik 2017 Eqn 16
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

    Parameters
    ----------
    opacities : dict
        Opacity dictionary of xarray.DataArray's
    temperature : ~astropy.units.Quantity
        Temperature value
    pressure : ~astropy.units.Quantity
        Pressure value
    lam : ~astropy.units.Quantity
        Wavelength bin centers
    m_bar : ~astropy.units.Quantity
        Mean molecular weight

    Returns
    -------
    k : ~astropy.units.Quantity
        Sum of opacities over all species
    sigma_scattering : ~astropy.units.Quantity
        Scattering cross section
    """
    sigma_scattering = rayleigh_H2(lam, m_bar) + rayleigh_He(lam, m_bar)

    if pressure.isscalar and temperature.isscalar: 
        pressure = u.Quantity([pressure])
        temperature = u.Quantity([temperature])

    ops = []

    interp_kwargs = dict(
        method='linear', 
        kwargs=dict(fill_value=0)
    )

    fastchem_mmr = chemistry(
        temperature, pressure, opacities.keys(), m_bar=m_bar
    )

    for species in opacities:

        interp_point = dict(
            pressure=xr.DataArray(pressure.to(u.bar).value, dims='z'),
        )

        if len(np.unique(opacities[species].temperature)) > 1:
            interp_point['temperature'] = xr.DataArray(
                temperature.value, dims='z'
            )

        opacity = fastchem_mmr[species][:, None] * opacities[species].interp(
            **interp_point, **interp_kwargs
        )
        ops.append(opacity)
    if len(ops) == 1:
        ops = ops[0].values.flatten()
    elif len(ops) > 1:
        ops = xr.concat(ops, 'opacities').sum('opacities').values.flatten()
    return ops * u.cm**2 / u.g + sigma_scattering, sigma_scattering


def load_example_opacity(grid, seed=42, scale_factor=20):
    """
    Load "example" opacity xarray. 
    
    This file function returns something compatible with 
    the output of ``binned_opacity``, so fake data can be 
    substituted for the real opacities during testing and 
    in the documentaiton.
    
    Parameters
    ----------
    grid : ~frei.Grid
        Grid object
    seed : int
        Random seed for numpy
    scale_factor : float
        Scale up/down synthetic opacities by this factor
        
    Returns
    -------
    op : dict
        Opacity tables for each species
    """
    np.random.seed(seed)
    simple_opacities = np.zeros(
        (grid.pressures.shape[0], grid.init_temperatures.shape[0],
         grid.lam.shape[0])
    )

    so = (
        # Broad infrared opacity
        np.exp(-0.5 * (grid.lam - 6 * u.um)**2 / (2 * u.um)**2) + 
        # Broad optical opacity
        0.8 * np.exp(-0.5 * (grid.lam - 0.3 * u.um)**2 / (0.5 * u.um)**2)
    )
    
    # Add a bunch of random absorption bands in the optical
    for amp, wl_micron in zip(
        np.random.uniform(low=0.1, high=0.2, size=15),
        np.random.uniform(low=0.5, high=1, size=15)
    ):
        so += amp * np.exp(
            -0.5 * (grid.lam - wl_micron * u.um)**2 / (0.005 * u.um)**2
        )
    
    # Add a few water-like absorption bands in the NIR
    for amp, wl_micron in zip(
        [0.22, 0.2, 0.18],
        np.logspace(np.log10(1.4), np.log10(2.7), 3)
    ):
        so += amp * np.exp(
            -0.5 * (grid.lam - wl_micron * u.um)**2 / (0.13 * u.um)**2
        )
    
    simple_opacities[:] += 5 * 10**(2.5 * (so.value - 0.4))
    simple_opacities *= scale_factor
    
    # Save this fake opacity grid to the water key in the opacity dictionary
    op = {
        "1H2-16O": xr.DataArray(
            simple_opacities, 
            dims=['pressure', 'temperature', 'wavelength'], 
            coords=dict(
                pressure=grid.pressures.to(u.bar).value, 
                temperature=grid.init_temperatures, 
                wavelength=grid.lam.to(u.um).value
            )
        ).drop_duplicates('temperature')
    }
    
    return op


def dace_download_molecule(
    isotopologue='48Ti-16O', linelist='Toto', 
    temperature_range=[500, 5000], pressure_range=[-6, 1.5], version=1
):
    from dace.opacity import Molecule
    os.makedirs('tmp', exist_ok=True)
    archive_name = isotopologue + '__' + linelist + '.tar.gz'
    Molecule.download(
        isotopologue, linelist, float(version),
        temperature_range, pressure_range,
        output_directory='tmp', output_filename=archive_name
    )
    
    return os.path.join('tmp', archive_name)


def dace_download_atom(
    element='Na', charge=0, linelist='Kurucz', 
    temperature_range=[500, 5000], pressure_range=[-8, 1.5], version=1
):
    from dace.opacity import Atom
    os.makedirs('tmp', exist_ok=True)
    archive_name = element + '__' + linelist + '.tar.gz'
    Atom.download(
        element, charge, linelist, float(version),
        temperature_range, pressure_range,
        output_directory='tmp', output_filename=archive_name
    )
    
    return os.path.join('tmp', archive_name)


def untar_bin_files(archive_name):
    def bin_files(members):
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".bin":
                yield tarinfo

    with tarfile.open(archive_name, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="tmp/.", members=bin_files(tar))


def get_opacity_dir_path_molecule(archive_name, isotopologue, linelist):
    return glob(os.path.join('tmp', isotopologue + '__' + linelist + "*e2b"))[0]


def get_opacity_dir_path_atom(linelist):
    return glob(os.path.join('tmp', linelist + "*e2b"))[0]


def opacity_dir_to_netcdf(opacity_dir, outpath):
    import xarray as xr

    temperature_grid = []
    pressure_grid = []

    for dirpath, dirnames, filenames in os.walk(opacity_dir): 
        for filename in filenames: 
            # Wavenumber points from range given in the file names
            temperature = int(filename.split('_')[3])
            sign = 1 if filename.split('_')[4][0] == 'p' else -1
            pressure = 10 ** (sign * float(filename.split('_')[4][1:].split('.')[0]) / 100)

            wl_start = int(filename.split('_')[1])
            wl_end = int(filename.split('_')[2])
            wlen = np.arange(wl_start, wl_end, 0.01)

            # Convert to micron
            wavelength = 1 / wlen / 1e-4

            unique_wavelengths = wavelength[1:][::-1]
            temperature_grid.append(temperature)
            pressure_grid.append(pressure)

    tgrid = np.sort(list(set(temperature_grid)))
    pgrid = np.sort(list(set(pressure_grid)))

    if len(pgrid) == 1:
        extrapolate_pgrid = True
        pgrid = np.concatenate([pgrid, 10**(-1*np.log10(pgrid))])
    else: 
        extrapolate_pgrid = False
    opacity_grid = np.zeros(
        (len(tgrid), len(pgrid), len(unique_wavelengths)), dtype='float32'
    )

    for dirpath, dirnames, filenames in os.walk(opacity_dir): 
        for filename in filenames: 

            opacity = np.fromfile(
                os.path.join(dirpath, filename), dtype=np.float32
            )[1:][::-1]

            # Wavenumber points from range given in the file names
            temperature = int(filename.split('_')[3])
            sign = 1 if filename.split('_')[4][0] == 'p' else -1
            pressure = 10 ** (sign * float(filename.split('_')[4][1:].split('.')[0]) / 100)

            temperature_ind = np.argmin(np.abs(tgrid - temperature))
            pressure_ind = np.argmin(np.abs(pgrid - pressure))

            opacity_grid[temperature_ind, pressure_ind, :] = opacity

    if extrapolate_pgrid:
        for dirpath, dirnames, filenames in os.walk(opacity_dir): 
            for filename in filenames: 

                opacity = np.fromfile(
                    os.path.join(dirpath, filename), dtype=np.float32
                )[1:][::-1]

                # Wavenumber points from range given in the file names
                temperature = int(filename.split('_')[3])
                # *Flip the sign for the extrapolated grid point in pressure*
                sign = -1 if filename.split('_')[4][0] == 'p' else 1
                pressure = 10 ** (sign * float(filename.split('_')[4][1:].split('.')[0]) / 100)

                temperature_ind = np.argmin(np.abs(tgrid - temperature))
                pressure_ind = np.argmin(np.abs(pgrid - pressure))

                opacity_grid[temperature_ind, pressure_ind, :] = opacity
            
    ds = xr.Dataset(
        data_vars=dict(
            opacity=(["temperature", "pressure", "wavelength"], 
                     opacity_grid)
        ),
        coords=dict(
            temperature=(["temperature"], tgrid),
            pressure=(["pressure"], pgrid),
            wavelength=unique_wavelengths
        )
    )
    
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    ds.to_netcdf(outpath if outpath.endswith(".nc") else outpath + '.nc', 
                 encoding={'opacity': {'dtype': 'float32', "zlib": True}})


def clean_up(bin_dir, archive_name): 
    os.remove(archive_name)
    shutil.rmtree(bin_dir)


def download_molecule(isotopologue, linelist):
    """
    Download molecular opacity data from DACE. 
    
    .. warning:: 
        This generates *very* large files. Only run this
        method if you have ~6 GB available per molecule.
        
    Parameters
    ----------
    isotopologue : str
        For example, "1H2-16O" for water.
    linelist : str
        For example, "POKAZATEL" for water.
    """
    archive_name = dace_download_molecule(isotopologue, linelist)
    untar_bin_files(archive_name)
    bin_dir = get_opacity_dir_path_molecule(
        archive_name, isotopologue, linelist
    )

    nc_path = os.path.join(
        os.path.expanduser('~'), '.frei', isotopologue + '__' +
        linelist + '.nc'
    )
    opacity_dir_to_netcdf(bin_dir, nc_path)
    clean_up(bin_dir, archive_name)


def download_atom(atom, charge, linelist):
    """
    Download atomic opacity data from DACE.

    .. warning:: 
        This generates *very* large files. Only run this
        method if you have ~6 GB available per molecule.
    
    Parameters
    ----------
    atom : str
        For example, "Na" for sodium.
    charge : int
        For example, 0 for neutral.
    linelist : str
        For example, "Kurucz".
    """
    archive_name = dace_download_atom(atom, charge, linelist)
    untar_bin_files(archive_name)
    bin_dir = get_opacity_dir_path_atom(linelist)
    
    nc_path = os.path.join(
        os.path.expanduser('~'), '.frei', atom + '_' + str(int(charge)) +
        '__' + linelist + '.nc'
    )
    opacity_dir_to_netcdf(bin_dir, nc_path)
    clean_up(bin_dir, archive_name)
