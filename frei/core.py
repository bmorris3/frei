import numpy as np
import astropy.units as u
from astropy.constants import m_p, G, sigma_sb
from specutils import Spectrum1D
from tqdm.auto import trange 

from .twostream import BB
from .tp import pressure_grid, temperature_grid
from .opacity import binned_opacity
from .phoenix import get_binned_phoenix_spectrum
from .plot import dashboard
from .twostream import emit, absorb

__all__ = [
    'dask_client',
    'Grid', 
    'Planet',
    'effective_temperature'
]


def dask_client(cluster_kwargs=dict(), client_kwargs=dict()):
    """
    Launch a local cluster, return the client.
    """
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(
        **cluster_kwargs
    )
    client = Client(cluster, **client_kwargs)
    return client


def wavelength_grid(min_micron=0.5, max_micron=10, n_bins=500, lam=None):
    """
    Compute a log-spaced wavelength grid.
    """
    if lam is None: 
        lam = np.logspace(np.log10(min_micron), np.log10(max_micron), n_bins) * u.um
    wl_bins = np.concatenate([
        [(lam.min() - (lam[1] - lam[0])).to(u.um).value], 
        lam.to(u.um).value]
    ) + (lam[1] - lam[0]).to(u.um).value / 2
    R = float(lam[lam.shape[0]//2] / (lam[lam.shape[0]//2 + 1] - lam[lam.shape[0]//2]))
    return lam, wl_bins, R


def F_TOA(lam, T_star=5800*u.K, f=2/3, a_rstar=float(0.03 * u.AU / u.R_sun)):
    """
    Compute the flux at the top of the atmosphere of the planet.
    """
    return (f * a_rstar ** -2 * 
        1 / (2 * np.pi) * 
        (np.pi * B_star(T_star, lam))
    ).to(u.erg/u.s/u.cm**3, u.spectral_density(lam))


def B_star(T_star, lam):
    """
    Compute the blackbody spectrum of the star
    """
    return BB(T_star)(lam)


class Planet(object): 
    """Container for planetary system information"""

    @u.quantity_input(
        m_bar=u.g, g=u.m/u.s**2, T_star=u.K
    )
    def __init__(self, a_rstar, m_bar, g, T_star, alpha):
        """
        Parameters
        ----------
        a_rstar : float
            Ratio of the semimajor axis of the orbit to the stellar radius
        m_bar : ~astropy.units.Quantity
            Mean molecular weight
        g : ~astropy.units.Quantity
            Surface gravity
        T_star : ~astropy.units.Quantity
            Stellar effective temperature
        alpha : float
            Number of scale heights in a mixing length
        """
        self.a_rstar = a_rstar
        self.m_bar = m_bar
        self.g = g
        self.T_star = T_star
        self.alpha = alpha

    @classmethod
    def from_hot_jupiter(cls):
        """
        Initialize a hot-Jupiter system with standard parameters:
        :math:`M=M_J`, :math:`R=R_J`, :math:`\\bar{m}=2.4`, :math:`g=g_J`,
        :math:`T_\\mathrm{eff}=5800` K.
        """
        g_jup = 1 * G * u.M_jup / u.R_jup**2
        return cls(
            a_rstar=float(0.03 * u.AU / u.R_sun), 
            m_bar=2.4*m_p,
            g=g_jup,
            T_star=5800 * u.K,
            alpha=1
        )

    
class Grid(object): 
    """
    Grid over temperatures, pressures and wavelengths.
    """
    @u.quantity_input(
        lam_min=u.um, lam_max=u.um, P_toa=u.bar, P_boa=u.bar,
        T_ref=u.K, P_ref=u.bar, lam=u.cm, pressures=u.bar, init_temperatures=u.K
    )
    def __init__(
        self, planet,
        lam=None, pressures=None, init_temperatures=None,
        # Wavelength grid:
        lam_min=0.5 * u.um, lam_max=10 * u.um, n_wl_bins=500,
        # Pressure grid: 
        P_toa=1e-6 * u.bar, P_boa=200 * u.bar, n_layers=30,
        # Initial temperature grid:
        T_ref=2300 * u.K, P_ref=0.1 * u.bar, alpha=0.1
    ):
        """
        If ``lam``, ``pressures``, or ``init_temperatures`` are None, 
        frei use the remaining keyword arguments to produce each grid.
        
        
        Parameters
        ----------
        planet : ~frei.Planet
            Planet object associated with this grid.
        lam : ~astropy.units.Quantity or None (optional)
            Wavelength grid
        pressures : ~astropy.units.Quantity or None (optional)
            Pressure grid
        init_temperatures : ~astropy.units.Quantity or None (optioonal)
            Initial temperature grid at each pressure
        lam_min : ~astropy.units.Quantity
            Minimum wavelength.
        lam_max : ~astropy.units.Quantity
            Maximum wavelength
        n_wl_bins : int
            Number of log-spaced bins in wavelength
        P_toa : ~astropy.units.Quantity
            Pressure at the top of the atmosphere
        P_boa : ~astropy.units.Quantity
            Pressure at the bottom of the atmosphere
        n_layers : int
            Number of log-spaced bins in pressure
        T_ref : ~astropy.units.Quantity
            Reference temperature at reference pressure
        P_ref : ~astropy.units.Quantity
            Reference pressure
        alpha : float (default = 0.1)
            Power law index of initial guess T-p profile 
        """
        self.planet = planet
        if lam is None:
            self.lam, self.wl_bins, self.R = wavelength_grid(
                min_micron=lam_min.to(u.um).value,
                max_micron=lam_max.to(u.um).value,
                n_bins=n_wl_bins
            )
        else: 
            self.lam, self.wl_bins, self.R = wavelength_grid(
                lam=lam
            )
            
        if pressures is None:
            self.pressures = pressure_grid(
                n_layers=n_layers, P_toa=np.log10(P_toa.to(u.bar).value),
                P_boa=np.log10(P_boa.to(u.bar).value)
            )
        else: 
            self.pressures = pressures
            
        if init_temperatures is None:
            self.init_temperatures = temperature_grid(
                self.pressures, T_ref, P_ref, alpha
            )
        else: 
            self.init_temperatures = init_temperatures
        
        self.opacities = None
        
    def __repr__(self): 
        return (
            f"<Grid in T=[{self.init_temperatures[0]:.0f}" + 
            f"...{self.init_temperatures[-1]:.0f}], " + 
            f"p=[{self.pressures[0]:.2g}...{self.pressures[-1]:.2g}], " + 
            f"lam=[{self.lam[0]}...{self.lam[-1]}]>"
        )
    
    def load_opacities(self, species=None, path=None, opacities=None, client=None):
        """
        Load opacity tables from path.

        Parameters
        ----------
        species : list or None (optional)
            List of species to load. If None, load all available.
        path : str
            Path passed to ~glob.glob to find opacity netCDF files.
        opacities : None or dict (optional)
            If opacities are already computed, simply pass them into the Grid
            object with this keyword argument.
        client : None or ~dask.distributed.client.Client
            Client for distributed dask computation on opacity tables

        Returns
        -------
        opacities : dict
            Opacity dictionary of xarray.DataArray's
        """
        if self.opacities is None and opacities is None:
            self.opacities = binned_opacity(
                self.init_temperatures,
                self.pressures, self.wl_bins, self.lam, client, 
                species
            )
        else: 
            self.opacities = opacities

        return self.opacities

    def emission_spectrum(self, n_timesteps=1, n_zero_crossings=2, convergence_dT=3*u.K):
        """
        Compute the emission spectrum for this grid.

        Parameters
        ----------
        n_timesteps : int
            Maximum number of timesteps to take towards radiative equilibrium
        convergence_thresh : ~astropy.units.Quantity
            When the maximum change in temperature between timesteps is less than
            ``convergence_thresh``, accept this timestep as "converged".
            
        Returns
        -------
        spec : specutils.Spectrum1D
            Emission spectrum
        final_temps : astropy.units.Quantity
            Final temperature grid
        temperature_history : astropy.units.Quantity
            Grid of temperatures with dimensions (n_layers, n_timesteps)
        dtaus : numpy.ndarray
            Change in optical depth in final iteration
        n_zero_crossings : int
            Number of changes in sign of ∆T in each layer before 
            considered converged
        convergence_dT : ~astropy.units.Quantity
            Alternatively, set the maximum change in temperature before 
            considered converged
        """
        if self.opacities is None:
            raise ValueError("Must load opacities before computing emission spectrum.")
        flux_unit = u.erg / u.s / u.cm**3
        F_toa = F_TOA(self.lam, T_star=self.planet.T_star, a_rstar=self.planet.a_rstar)
        final_temps = self.init_temperatures.copy()
        n_layers, n_wavelengths = len(self.pressures), len(self.lam)
        fluxes_down = np.zeros((n_layers, n_wavelengths)) * flux_unit
        fluxes_up = np.zeros((n_layers, n_wavelengths)) * flux_unit
        temp_hists = []
        max_dTs = []
        timestep_iterator = trange(n_timesteps)
        
        for i in timestep_iterator:
        
            fluxes_up, fluxes_down, final_temps, temperature_history_emit, _, dT = emit(
                opacities=self.opacities, 
                temperatures=final_temps, 
                pressures=self.pressures, 
                lam=self.lam, 
                F_TOA=F_toa, 
                g=self.planet.g, 
                m_bar=self.planet.m_bar,
                n_timesteps=1,
                alpha=self.planet.alpha,
                fluxes_up=fluxes_up, fluxes_down=fluxes_down
            )

            fluxes_up, fluxes_down, final_temps, temperature_history_absorb, _, dT = absorb(
                opacities=self.opacities, 
                temperatures=final_temps, 
                pressures=self.pressures, 
                lam=self.lam, 
                F_TOA=F_toa, 
                g=self.planet.g, 
                m_bar=self.planet.m_bar,
                n_timesteps=1,
                alpha=self.planet.alpha, 
                fluxes_up=fluxes_up, fluxes_down=fluxes_down
            )
            
            max_dT = np.abs(dT).max()
            # Check for convergence with 6 sign flips in temperature history diff
            temp_hists.append(temperature_history_absorb)
            max_dTs.append(max_dT)

            temp_hist = np.hstack(temp_hists)
            temp_hist = temp_hist.T[temp_hist[0] != 0].T            
            diffs = np.diff(temp_hist.value.T, axis=0)
            conv = (np.count_nonzero(
                np.sign(diffs[1:]) != np.sign(diffs[:-1]), axis=0
            ) > n_zero_crossings) | (np.abs(dT) < convergence_dT)
            timestep_iterator.set_description(
                f"max|∆T|={max_dT:.1f}; conv = {np.count_nonzero(conv)}/{n_layers}"
            )

            if np.all(conv):
                break
        
        temp_hist = np.hstack(temp_hists)
        temp_hist = temp_hist.T[temp_hist[0] != 0].T
        
        fluxes_up, fluxes_down, final_temps, _, dtaus, dT = emit(
            opacities=self.opacities, 
            temperatures=final_temps, 
            pressures=self.pressures, 
            lam=self.lam, 
            F_TOA=F_toa, 
            g=self.planet.g, 
            m_bar=self.planet.m_bar,
            n_timesteps=1,
            fluxes_up=fluxes_up, fluxes_down=fluxes_down
        )
        
        return (
            Spectrum1D(flux=fluxes_up[-1], spectral_axis=self.lam), 
            final_temps, temp_hist, dtaus
        )
    
    def emission_dashboard(self, spec, final_temps, temperature_history, dtaus,
                           T_eff=None, plot_phoenix=True, cache=False):
        """
        Produce the "dashboard" plot with the outputs from ``emission_spectrum``.

        Parameters
        ----------
        spec : ~specutils.Spectrum1D
            Emission spectrum
        final_temps : ~astropy.units.Quantity
            Final temperature grid
        temperature_history : ~astropy.units.Quantity
            Grid of temperatures with dimensions (n_layers, n_timesteps)
        dtaus : ~np.ndarray
            Change in optical depth in final iteration
        T_eff : ~astropy.units.Quantity or None
            If not None, give the effective temperature of the PHOENIX model
            to plot in comparison, otherwise compute it on the fly.
        plot_phoenix : bool
            If True, plot the corresponding PHOENIX model
        cache : bool
            Cache the PHOENIX model spectrum if ``plot_phoenix`` is True.
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects.
        """
        if plot_phoenix:
            if T_eff is None:
                T_eff = effective_temperature(self, spec, dtaus, final_temps)

            phoenix_lowres_padded = get_binned_phoenix_spectrum(
                T_eff, self.planet.g, self.wl_bins, self.lam, cache=cache
            )
        else:
            flux_unit = u.erg/u.cm**3/u.s
            phoenix_lowres_padded = np.zeros(len(self.lam)) * flux_unit
        
        fig, ax = dashboard(
            self.lam, spec.flux, phoenix_lowres_padded, dtaus, 
            self.pressures, final_temps, temperature_history, self.opacities
        )
        
        return fig, ax


def effective_temperature_milne(grid, spec, dtaus, final_temps):
    """
    Estimate photosphere temperature from Milne's solution (tau ~ 2/3).
    """
    pressure_milne = np.ones_like(grid.lam.value)

    for i in range(dtaus.shape[1]):
        pressure_milne[i] = np.interp(
            2/3, np.exp(-dtaus[:, i]), grid.pressures
        ).to(u.bar).value

    temperature_milne = np.interp(
        np.average(
            pressure_milne,
            weights=spec.flux.to(u.erg/u.s/u.cm**2,
                                 u.spectral_density(grid.lam)).value
        ),
        grid.pressures[::-1].to(u.bar).value, final_temps[::-1]
    )
    return temperature_milne


def effective_temperature_planck(grid, spec):
    """
    Use the Stefan-Boltzmann law to invert the emitted flux for the
    effective temperature.
    """
    bol_flux = np.trapz(spec.flux, grid.lam)
    return ((bol_flux / sigma_sb) ** (1/4)).decompose()


def effective_temperature(grid, spec, dtaus, final_temps):
    """
    Compute effective temperature of an atmosphere given the outputs
    from ``emit``.

    This is the mean of the effective temperatures computed from
    Milne's solution and the Stefan-Boltzmann law.

    Parameters
    ----------
    grid : ~frei.Grid
        Wavelength, pressure and temperature grid
    spec : ~specutils.Spectrum1D
        Emission spectrum
    dtaus : ~numpy.ndarray
        Change in optical depth in final iteration
    final_temps : ~astropy.units.Quantity
        Temperature grid in the final iteration
    """
    return u.Quantity([
        effective_temperature_milne(grid, spec, dtaus, final_temps),
        effective_temperature_planck(grid, spec)
    ]).mean()
