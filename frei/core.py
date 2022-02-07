import numpy as np
import astropy.units as u
from astropy.constants import m_p, G
from specutils import Spectrum1D
from dask.distributed import Client, LocalCluster

from .twostream import BB
from .tp import pressure_grid, temperature_grid
from .opacity import binned_opacity
from .phoenix import get_binned_phoenix_spectrum
from .plot import dashboard
from .twostream import emit

__all__ = [
    'dask_client', 
    'wavelength_grid', 
    'F_TOA', 
    'Grid', 
    'Planet', 
]

def dask_client(memory_limit='20 GiB'):
    """
    Launch a local cluster, return the client.
    """
    cluster = LocalCluster(
        memory_limit=memory_limit
    )
    client = Client(cluster)
    return client


def wavelength_grid(min_micron=0.5, max_micron=10, n_bins=500):
    """
    Compute a wavelength grid. 
    """
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
    def __init__(self, a_rstar, m_bar, g, T_star):
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
        """
        self.a_rstar = a_rstar
        self.m_bar = m_bar
        self.g = g
        self.T_star = T_star

    @classmethod
    def from_hot_jupiter(cls):
        """
        Initialize a hot-Jupiter system with standard parameters:
        :math:`M=M_J`, :math:`R=R_J`, :math:`\mu=2.4`, :math:`g=g_J`,
        :math:`T_\mathrm{eff}=5800` K.
        """
        g_jup = 1 * G * u.M_jup / u.R_jup**2
        return cls(
            a_rstar=float(0.03 * u.AU / u.R_sun), 
            m_bar=2.4*m_p,
            g=g_jup,
            T_star=5800 * u.K
        )

    
class Grid(object): 
    """
    Grid over temperatures, pressures and wavelengths.
    """
    def __init__(
        self, planet,
        # Wavelength grid:
        lam_min=0.5, lam_max=10, n_wl_bins=500,
        # Pressure grid: 
        n_layers=30, P_toa=-6, P_boa=1.1,
        # Initial temperature grid:
        T_ref=2300 * u.K, P_ref = 0.1 * u.bar
    ):
        """
        Parameters
        ----------
        planet : ~frei.Planet
            Planet object associated with this grid.
        lam_min : float
            Minimum wavelength in units of microns.
        lam_max : float
            Maximum wavelength in units of microns.
        n_wl_bins : int
            Number of log-spaced bins in wavelength
        n_layers : int
            Number of log-spaced bins in pressure
        P_toa : float
            Pressure at the top of the atmosphere in log10(bar)
        P_boa : float
            Pressure at the bottom of the atmosphere in log10(bar)
        T_ref : ~astropy.units.Quantity
            Reference temperature at reference pressure
        P_ref : ~astropy.units.Quantity
            Reference pressure
        """
        self.planet = planet
        self.lam, self.wl_bins, self.R = wavelength_grid(
            min_micron=lam_min, max_micron=lam_max, n_bins=n_wl_bins
        )
        
        self.pressures = pressure_grid(
            n_layers=n_layers, P_toa=P_toa, P_boa=P_boa
        )

        self.init_temperatures = temperature_grid(
            self.pressures, T_ref, P_ref
        )
        
        self.opacities = None
        
    def __repr__(self): 
        return (
            f"<Grid in T=[{self.init_temperatures[0]:.0f}" + 
            f"...{self.init_temperatures[-1]:.0f}], " + 
            f"p=[{self.pressures[0]:.2g}...{self.pressures[-1]:.2g}], " + 
            f"lam=[{self.lam[0]}...{self.lam[-1]}]>"
        )
    
    def load_opacities(self, path='tmp/*.nc', opacities=None, client=None):
        """
        Load opacity tables from path.

        Parameters
        ----------
        path : str
            Path passed to ~glob.glob to find opacity netCDF files.
        opacities : None or dict (optional)
            If opacities are already computed, simply pass them into the Grid
            object with this keyword argument.
        client : None or ~dask.distributed.client.Client
            Client for distributed dask computation on opacity tables
        """
        if self.opacities is None and opacities is None:
            self.opacities = binned_opacity(
                path, self.init_temperatures,
                self.pressures, self.wl_bins, self.lam, client
            )
        else: 
            self.opacities = opacities

    def emission_spectrum(self, n_timesteps=50):
        """
        Compute the emission spectrum for this grid.

        Parameters
        ----------
        n_timesteps : int
            Maximum number of timesteps to take towards radiative equilibrium

        Returns
        -------
        spec : specutils.Spectrum1D
            Emission spectrum
        final_temps : ~astropy.units.Quantity
            Final temperature grid
        temperature_history : ~astropy.units.Quantity
            Grid of temperatures with dimensions (n_layers, n_timesteps)
        dtaus : ~numpy.ndarray
            Change in optical depth in final iteration
        """
        if self.opacities is None:
            raise ValueError("Must load opacities before computing emission spectrum.")
        
        F_toa = F_TOA(self.lam, T_star=self.planet.T_star)
        F_2_up, final_temps, temperature_history, dtaus = emit(
            opacities=self.opacities, 
            temperatures=self.init_temperatures, 
            pressures=self.pressures, 
            lam=self.lam, 
            F_TOA=F_toa, 
            g=self.planet.g, 
            m_bar=self.planet.m_bar,
            n_timesteps=n_timesteps
        )
        return (
            Spectrum1D(flux=F_2_up, spectral_axis=self.lam), 
            final_temps, temperature_history, dtaus
        )
    
    def emission_dashboard(self, spec, final_temps, temperature_history, dtaus, T_eff=2400*u.K):
        """
        Produce the "daskboard" plot with the outputs from ``emit``.

        Parameters
        ----------
        spec : specutils.Spectrum1D
            Emission spectrum
        final_temps : ~astropy.units.Quantity
            Final temperature grid
        temperature_history : ~astropy.units.Quantity
            Grid of temperatures with dimensions (n_layers, n_timesteps)
        dtaus : ~numpy.ndarray
            Change in optical depth in final iteration
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects.
        """
        phoenix_lowres_padded = get_binned_phoenix_spectrum(
            T_eff, self.planet.g, self.wl_bins, self.lam
        )
        
        fig, ax = dashboard(
            self.lam, spec.flux, phoenix_lowres_padded, dtaus, 
            self.pressures, final_temps, temperature_history
        )
        
        return fig, ax
