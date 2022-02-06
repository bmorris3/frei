import matplotlib.pyplot as  plt
import numpy as np
from astropy.constants import h, c, k_B
from matplotlib.gridspec import GridSpec
import astropy.units as u
from astropy.visualization import quantity_support

from .chemistry import chemistry


def dashboard(lam, F_2_up, phoenix_lowres_padded, dtaus, pressures, temps, temperature_history):
    flux_unit = u.erg/u.cm**3/u.s

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, figure=fig)

    ax = [fig.add_subplot(ax) for ax in [gs[0, :], gs[1, 0], gs[1, 1], gs[1, 2]]]

    with quantity_support():
        ax[0].loglog(
            lam, phoenix_lowres_padded * flux_unit, color='C1', label='PHOENIX'
        )
        ax[0].semilogx(lam, F_2_up, color='C0', label='frei')

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
    cax = ax[1].pcolormesh(lg, pg, cf[::-1], cmap=plt.cm.Greys, shading='auto')#, vmin=0, vmax=1)
    plt.colorbar(cax, ax=ax[1])
    ax[1].set_yscale('log')
    ax[1].invert_yaxis()
    ax[1].set(
        xlabel='Wavelength [$\mu$m]', ylabel='Pressure [bar]', title='Contrib Func'
    )
    ax[0].set(
        xlabel='Wavelength [$\mu$m]', title='Emission spectrum', 
        ylim=[8e10, 3e13], xlim=[0.5, 10] # #fluxes_upwards[-1].min(),  # xlim=[lam.min(), lam.max()],
    )
    ax[1].set_xscale('log')

    cmap = plt.cm.winter_r
    for i in range(temperature_history.shape[1]):
        color = cmap(i / temperature_history.shape[1])
        if np.all(temperature_history[:, i] != 0):
            ax[2].semilogy(temperature_history[:-1, i], pressures[:-1], c=color, alpha=0.3);
    ax[2].semilogy(temps[:-1], pressures[:-1], '-', color='k', lw=3)
    ax[2].invert_yaxis()
    ax[2].annotate("Initial", (0.1, 0.18), color=cmap(0), xycoords='axes fraction')
    ax[2].annotate("Final", (0.1, 0.1), xycoords='axes fraction')
    ax[2].set(
        xlabel='Temperature [K]', ylabel='Pressure [bar]'
    )

    fastchem_mmr, fastchem_vmr = chemistry(temps[:-1], pressures[:-1], return_vmr=True)
    for species in fastchem_vmr:
        ax[3].semilogy(
            np.log10(fastchem_vmr[species]), pressures[:-1], 
            label=species.replace('2', '$_2$'), lw=2
        )
    ax[3].legend()
    ax[3].invert_yaxis()
    ax[3].set(
        xlabel='log(VMR)', ylabel='Pressure [bar]',
        title='Chemistry (FastChem)'
    )

    for axis in ax: 
        for sp in ['right', 'top']:
            axis.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig, ax
    # fig.savefig('plots/demo00.png', bbox_inches='tight', dpi=200)