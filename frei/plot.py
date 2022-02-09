import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.units as u
from astropy.constants import h, c, k_B
from astropy.visualization import quantity_support

from .chemistry import chemistry
from .opacity import kappa

__all__ = [
    'dashboard'
]


def dashboard(
    lam, F_2_up, binned_phoenix_spectrum, dtaus,
    pressures, temps, temperature_history, opacities
):
    """
    Generate a dashboard plot.

    Parameters
    ----------
    lam : ~astropy.units.Quantity
        Wavelength grid
    F_2_up : ~astropy.units.Quantity
        Emission spectrum
    binned_phoenix_spectrum : ~astropy.units.Quantity
        Binned PHOENIX spectrum
    dtaus : list of lists, or ~numpy.ndarray
        Change in optical depth
    pressures : ~astropy.units.Quantity
        Pressure grid
    temps : ~astropy.units.Quantity
        Final temperatures after iteration for radiative equilibrium
    temperature_history : ~astropy.units.Quantity
        Grid of temperatures for each timestep and pressure
    opacities : dict
        Opacity dictionary of xarray.DataArray's

    Returns
    -------
    fig, ax : ~matplotlib.axes.Figure, ~matplotlib.axes.Axes
    """
    from .opacity import iso_to_species
    flux_unit = u.erg/u.cm**3/u.s

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 4, figure=fig)

    ax = [fig.add_subplot(ax)
          for ax in [gs[0, :], gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]]

    with quantity_support():
        if np.any(binned_phoenix_spectrum.value != 0):
            ax[0].loglog(
                lam, binned_phoenix_spectrum, color='C1', label='PHOENIX'
            )
        ax[0].loglog(lam, F_2_up.to(flux_unit), color='C0', label='frei')

    ax[0].legend()
    tau = np.cumsum(dtaus[::-1], axis=0)
    nus = lam.to(u.cm**-1, u.spectral())
    hcperk = h * c / k_B

    dlogP = (np.log10(pressures.max().to(u.bar).value) - 
        np.log10(pressures.min().to(u.bar).value)
    ) / (len(pressures) - 1)
    k = 10 ** -dlogP
    dParr = (1 - k) * pressures

    cf = (
        np.exp(-tau) * np.array(dtaus)[::-1] * 
        (pressures[::-1, None] / dParr[::-1, None]) * 
        nus**3 / np.expm1(hcperk * nus / 
                          temps[::-1, None]))

    cf /= np.sum(cf, axis=0)

    lg, pg = np.meshgrid(lam.value, pressures.value)
    cax = ax[1].pcolormesh(lg, pg, cf[::-1], cmap=plt.cm.Greys, shading='auto')
    plt.colorbar(cax, ax=ax[1])
    ax[1].set_yscale('log')
    ax[1].invert_yaxis()
    ax[1].set(
        xlabel=r'Wavelength [$\mu$m]', ylabel='Pressure [bar]',
        title='Contrib Func',
        xlim=[lam.value.min(), lam.value.max()], 
        ylim=[pressures.value.max(), pressures.value.min()]
    )
    ax[0].set(
        xlabel=r'Wavelength [$\mu$m]', title='Emission spectrum',
    )
    ax[1].set_xscale('log')

    cmap = plt.cm.winter_r
    for i in range(temperature_history.shape[1]):
        color = cmap(i / temperature_history.shape[1])
        if np.all(temperature_history[:, i] != 0):
            ax[2].semilogy(temperature_history[:-1, i], pressures[:-1],
                           c=color, alpha=0.3)
    ax[2].semilogy(temps[:-1], pressures[:-1], '-', color='k', lw=3)
    ax[2].invert_yaxis()
    ax[2].annotate("Initial", (0.1, 0.18), color=cmap(0),
                   xycoords='axes fraction')
    ax[2].annotate("Final", (0.1, 0.1), xycoords='axes fraction')
    ax[2].set(
        xlabel='Temperature [K]', ylabel='Pressure [bar]',
        ylim=ax[1].get_ylim()
    )

    fastchem_mmr, fastchem_vmr = chemistry(
        temps[:-1], pressures[:-1], opacities.keys(), return_vmr=True
    )

    for isotopologue in fastchem_vmr:
        
        species_name = iso_to_species(isotopologue)
        ax[3].semilogy(
            np.log10(fastchem_vmr[isotopologue]), pressures[:-1],
            label=species_name.replace('2', '$_2$'), lw=2
        )
    ax[3].legend()
    ax[3].invert_yaxis()
    ax[3].set(
        xlabel='log(VMR)', ylabel='Pressure [bar]',
        title='Chemistry (FastChem)',
        ylim=ax[1].get_ylim()
    )

    k, sigma_scattering = kappa(
        opacities, np.interp(1 * u.bar, pressures, temps), 1 * u.bar, lam
    )
    with quantity_support():
        ax[4].loglog(lam, k.to(u.cm ** 2 / u.g), label='Total')
        ax[4].loglog(lam, sigma_scattering.to(u.cm ** 2 / u.g), label='Scattering')
    ax[4].set(
        xlabel=r'Wavelength [$\mu$m]', ylabel='Opacity [cm$^2$ g$^{-1}$]'
    )
    ax[4].legend()
    for axis in ax: 
        for sp in ['right', 'top']:
            axis.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig, ax
