import astropy.units as u
import numpy as np
from astropy.constants import k_B, m_p, h, c, sigma_sb
from tqdm.auto import trange

from .opacity import kappa

__all__ = [
    'propagate_fluxes',
    'emit', 'absorb'
]

flux_unit = u.erg / u.s / u.cm ** 3


def bolometric_flux(flux, lam):
    """
    Compute bolometric flux from wavelength-dependent flux
    """
    return np.trapz(flux, lam)


def delta_t_i(p_1, p_2, T_1, T_2, delta_F_i_dz, g, m_bar=2.4 * m_p, n_dof=5):
    """
    Timestep in iteration for radiative equilibrium.

    Follows Equations 27-28 of Malik et al. (2017).
    """
    dz = delta_z_i(T_1, p_1, p_2, g, m_bar)
    # Malik 2017 Eqn 28
    
    if (delta_F_i_dz * dz).value != 0:
        f_i_pre = 1e5 / (abs(delta_F_i_dz * dz) / (u.erg/u.cm**2/u.s))**0.9
    else: 
        f_i_pre = 1
    # Malik 2017 Eqn 27
    dt_radiative = c_p(m_bar=m_bar, n_dof=n_dof) * p_1 / sigma_sb / g / T_1 ** 3
    
    d_gamma = delta_gamma(T_1, T_2, p_1, p_2, g, m_bar=m_bar, n_dof=n_dof)
    if d_gamma > 0 * u.K / u.km:
        dt_convective = (T_1 / g / d_gamma) ** 0.5
        return f_i_pre * min(dt_radiative, dt_convective)
    return f_i_pre * dt_radiative


def BB(temperature):
    """
    Compute the blackbody flux

    Parameters
    ----------
    temperature : ~astropy.units.Quantity
        Temperature of the blackbody

    Returns
    -------
    bb : function
        Returns a function which takes the wavelength as an
        `~astropy.units.Quantity` and returns the Planck flux
    """
    # h = 6.62607015e-34  # J s
    # c = 299792458.0  # m/s
    # k_B = 1.380649e-23  # J/K
    return lambda wavelength: (
            2 * h * c ** 2 / np.power(wavelength, 5) /
            np.expm1(h * c / (wavelength * k_B * temperature))
    )


def E(omega_0, g_0):
    """
    Improved two-stream equation correction term.

    From Deitrick et al. (2020) Equation 19.

    Parameters
    ----------
    omega_0 : float or ~numpy.ndarray
        Single-scattering albedo
    g_0 : float or ~numpy.ndarray
        Scattering asymmetry factor

    Returns
    -------
    corr : ~numpy.ndarray
        Correction term, E(omega_0, g_0).
    """
    # Deitrick (2020) Eqn 19
    return np.where(
        omega_0 > 0.1,
        1.225 - 0.1582 * g_0 - 0.1777 * omega_0 - 0.07465 *
        g_0 ** 2 + 0.2351 * omega_0 * g_0 - 0.05582 * omega_0 ** 2,
        1
    )


def propagate_fluxes(
        lam, F_1_up, F_2_down, T_1, T_2, delta_tau, omega_0=0, g_0=0, eps=0.5
):
    """
    Compute fluxes up and down using the improved two-stream equations.

    The transmission function is taken from Deitrick et al. (2020) Equation B2.

    The two stream equations are taken from Malik et al. (2017)
    (see Equation 15), with corrections from Dietrick et al. (2022)
    (see Appendix B).

    Parameters
    ----------
    lam : ~astropy.units.Quantity
        Wavelength grid
    F_1_up : ~astropy.units.Quantity
        Flux up into layer 1
    F_2_down : ~astropy.units.Quantity
        Flux down into layer 2
    T_1 : ~astropy.units.Quantity
        Temperature in layer 1
    T_2 : ~astropy.units.Quantity
        Temperature in layer 2
    delta_tau : ~numpy.ndarray
        Change in optical depth
    omega_0 : ~numpy.ndarray or float
        Single scattering albedo
    g_0 : ~numpy.ndarray or float
        Scattering asymmetry factor
    eps : float
        First Eddington coefficient (Heng et al. 2014)

    Returns
    -------
    F_2_up, F_1_down : ~astropy.units.Quantity
        Fluxes outgoing to layer 2, and incoming to layer 1
    """
    omega_0 = omega_0.flatten()
    delta_tau = delta_tau.flatten()
    
    # Deitrick 2020 Equation B2
    T = np.exp(-2 * (E(omega_0, g_0) * (E(omega_0, g_0) - omega_0) *
                     (1 - omega_0 * g_0)) ** 0.5 * delta_tau)

    # Malik 2017 Equation 13
    zeta_plus = 0.5 * (1 + ((E(omega_0, g_0) - omega_0) / E(omega_0, g_0) /
                            (1 - omega_0 * g_0)) ** 0.5)
    zeta_minus = 0.5 * (1 - ((E(omega_0, g_0) - omega_0) / E(omega_0, g_0) /
                             (1 - omega_0 * g_0)) ** 0.5)

    # Malik 2017 Equation 12
    chi = zeta_minus ** 2 * T ** 2 - zeta_plus ** 2
    xi = zeta_plus * zeta_minus * (1 - T ** 2)
    psi = (zeta_minus ** 2 - zeta_plus ** 2) * T
    pi = np.pi * (1 - omega_0) / (E(omega_0, g_0) - omega_0)

    B1 = BB(T_1)(lam)
    B2 = BB(T_2)(lam)

    # Malik 2017 Equation 5
    Bprime = (B1 - B2) / delta_tau

    # Deitrick 2022 Eqn B4
    F_2_up = (
        1 / chi * (
            psi * F_1_up - xi * F_2_down +
            pi * (B2 * (chi + xi) - psi * B1 +
            Bprime / (2 * E(omega_0, g_0) * (1 - omega_0 * g_0)) *
            (chi - psi - xi))
        )
    )
    F_1_down = (
        1 / chi * (
            psi * F_2_down - xi * F_1_up +
            pi * (B1 * (chi + xi) - psi * B2 +
            Bprime / (2 * E(omega_0, g_0) * (1 - omega_0 * g_0)) *
            (xi + psi - chi))
        )
    )
    return F_2_up, F_1_down


def delta_z_i(temperature_i, pressure_i, pressure_ip1, g, m_bar=2.4 * m_p):
    """
    Change in height in the atmosphere from bottom to top of a layer.

    Malik et al. (2017) Equation 18
    """
    return ((k_B * temperature_i) / (m_bar * g) *
            np.log(pressure_i / pressure_ip1))


def div_bol_net_flux(
    F_ip1_u, F_ip1_d, F_i_u, F_i_d, temperature_i, temperature_ip1, pressure_i, pressure_ip1,
    g, m_bar=2.4 * m_p, n_dof=5, alpha=1
):
    """
    Divergence of the bolometric net flux.

    Defined in Malik et al. (2017) Equation 23.
    """
    delta_F_rad = (F_ip1_u - F_ip1_d) - (F_i_u - F_i_d)
    
    delta_F_conv = convective_flux(temperature_i, temperature_ip1, 
                                   pressure_i, pressure_ip1, g, 
                                   m_bar=m_bar, n_dof=n_dof, alpha=alpha)
    dz = delta_z_i(temperature_i, pressure_i, pressure_ip1, g, m_bar)
    return (delta_F_rad + delta_F_conv) / dz, dz


def delta_temperature(
        div, p_1, p_2, T_1, delta_t_i, g, m_bar=2.4 * m_p, n_dof=5
):
    """
    Change in temperature in each layer after timestep for radiative equilibrium

    Defined in Malik et al. (2017) Equation 24
    """
    return (1 / rho_p(p_1, p_2, T_1, g, m_bar) / 
            c_p(m_bar, n_dof) * div * delta_t_i)


def c_p(m_bar=2.4 * m_p, n_dof=5):
    """
    Heat capacity, Malik et al. (2017) Equation 25
    """
    return (2 + n_dof) / (2 * m_bar) * k_B


def delta_tau_i(kappa_i, p_1, p_2, g):
    """
    Contribution to optical depth from layer i, Malik et al. (2017) Equation 19
    """
    return (p_1 - p_2) / g * kappa_i


def rho_p(p_1, p_2, T_1, g, m_bar=2.4 * m_p):
    """
    Local density.
    """
    return ((p_1 - p_2) / g) / delta_z_i(T_1, p_1, p_2, g, m_bar)


def gamma(temperature_i, temperature_ip1, pressure_i, pressure_ip1, g, m_bar=2.4 * m_p):
    """
    Change in temperature with height
    """
    return (
        (temperature_i - temperature_ip1) / 
        delta_z_i(
            temperature_i, pressure_i, pressure_ip1, g, m_bar=m_bar
        )
    )


def gamma_adiabatic(g, m_bar=2.4 * m_p, n_dof=5):
    return g / c_p(m_bar=m_bar, n_dof=n_dof)


def delta_gamma(
    temperature_i, temperature_ip1, pressure_i, pressure_ip1, g, 
    m_bar=2.4 * m_p, n_dof=5
):
    dg = (
        gamma(temperature_i, temperature_ip1, 
              pressure_i, pressure_ip1, g, m_bar=m_bar) - 
        gamma_adiabatic(g, m_bar=m_bar, n_dof=n_dof)
    )
    return dg


def mixing_length(T_1, g, alpha=1, m_bar=2.4*m_p):
    return alpha * k_B * T_1 / (m_bar * g)


def convective_flux(
    temperature_i, temperature_ip1, pressure_i, pressure_ip1, g, 
    m_bar=2.4 * m_p, n_dof=5, alpha=1
):
    rho = rho_p(pressure_i, pressure_ip1, temperature_i, g, m_bar=m_bar)
    cp = c_p(m_bar=m_bar, n_dof=n_dof)
    lmix = mixing_length(temperature_i, g, alpha, m_bar)
    delta_g = delta_gamma(
        temperature_i, temperature_ip1, pressure_i, pressure_ip1, 
        g, m_bar=m_bar, n_dof=n_dof
    )
    
    if delta_g > 0 * u.K / u.km: 
        return rho * cp * lmix**2 * (g / temperature_i)**0.5 * delta_g**1.5
    return 0 * flux_unit * u.cm


def emit(
    opacities, temperatures, pressures, lam, F_TOA, g, m_bar=2.4 * m_p,
    n_timesteps=50, convergence_thresh=10 * u.K, alpha=1, fluxes_up=None,
    fluxes_down=None
):
    """
    Compute emission spectrum.

    Parameters
    ----------
    opacities : dict
        Opacity database binned to wavelength grid.
    temperatures : ~astropy.units.Quantity
        Temperature grid
    pressures : ~astropy.units.Quantity
        Pressure grid
    lam : ~astropy.units.Quantity
        Wavelength grid
    F_TOA : ~astropy.units.Quantity
        Flux at the top of the atmosphere
    g : ~astropy.units.Quantity
        Surface graivty
    m_bar : ~astropy.units.Quantity
        Mean molecular weight
    n_timesteps : int
        Maximum number of timesteps in iteration for radiative equilibrium
    convergence_thresh : ~astropy.units.Quantity
        When the maximum change in temperature between timesteps is less than
        ``convergence_thresh``, accept this timestep as "converged".

    Returns
    -------
    F_2_up : ~astropy.units.Quantity
        Outgoing flux
    final_temps : ~astropy.units.Quantity
        Final temperature grid
    temperature_history : ~astropy.units.Quantity
        Grid of temperatures with dimensions (n_layers, n_timesteps)
    dtaus : ~numpy.ndarray
        Change in optical depth in final iteration
    """
    n_layers = len(pressures)
    n_wavelengths = len(lam)

    if fluxes_up is None: 
        fluxes_up = np.zeros((n_layers, n_wavelengths)) * flux_unit

    if fluxes_down is None:
        fluxes_down = np.zeros((n_layers, n_wavelengths)) * flux_unit
        fluxes_down[-1] = F_TOA
    
    # from bottom of the atmosphere
    temperature_history = np.zeros((n_layers, n_timesteps + 1)) * u.K
    temperature_history[:, 0] = temperatures.copy()
    
    if n_timesteps > 1:
        timestep_iterator = trange(n_timesteps)

    else:
        timestep_iterator = np.arange(n_timesteps)

    for j in timestep_iterator:
        dtaus = [[1, ] * n_wavelengths]
        temps = temperature_history[:, j]
        temperature_changes = np.zeros(n_layers) * u.K

        for i in np.arange(1, n_layers):
            
            if i == n_layers - 1:
                p_2 = pressures[i] * pressures[-2] / pressures[-3]
                T_2 = temps[i]
            else: 
                p_2 = pressures[i + 1]
                T_2 = temps[i + 1]
            
            p_1 = pressures[i]
            T_1 = temps[i]

            k, sigma_scattering = kappa(
                opacities, T_1, p_1, lam, m_bar
            )
            delta_tau = delta_tau_i(
                k, p_1, p_2, g
            ).to(u.dimensionless_unscaled).value
            dtaus.append(delta_tau)
            # Single scattering albedo, Deitrick (2020) Eqn 17
            omega_0 = (
                sigma_scattering / (sigma_scattering + k)
            ).to(u.dimensionless_unscaled).value
            if i < n_layers - 1:
                F_2_down = fluxes_down[i + 1]
            else: 
                F_2_down = F_TOA
            F_1_up = fluxes_up[i]

            F_2_up, F_1_down = propagate_fluxes(
                lam,
                F_1_up, F_2_down, T_1, T_2,
                delta_tau,
                omega_0=omega_0, g_0=0
            )

            if i < n_layers - 1: 
                fluxes_up[i + 1] = F_2_up
            fluxes_down[i] = F_1_down

            delta_F_i_dz, dz = div_bol_net_flux(
                bolometric_flux(F_2_up, lam), bolometric_flux(F_2_down, lam),
                bolometric_flux(F_1_up, lam), bolometric_flux(F_1_down, lam),
                T_1, T_2, p_1, p_2, g, alpha=alpha, m_bar=m_bar
            )
            dt = delta_t_i(p_1, p_2, T_1, T_2, delta_F_i_dz, g, m_bar=m_bar)

            temperature_changes[i] = delta_temperature(
                delta_F_i_dz, p_1, p_2, T_1, dt, g
            ).decompose()
        dT = u.Quantity(temperature_changes)
        temperature_history[:, j + 1] = temps - dT
        converged = np.all(np.abs(dT).max() < convergence_thresh)
        if n_timesteps > 1:
            timestep_iterator.set_description(
                f"max|∆T|={np.abs(dT).max():.1f}"
            )

            # Stop iterating if T-p profile changes by <convergence_thresh
            if converged:
                break

    return (
        fluxes_up, fluxes_down, temperature_history[:, j + 1], 
        temperature_history, np.array(dtaus), dT
    )


def absorb(
    opacities, temperatures, pressures, lam, F_TOA, g, m_bar=2.4 * m_p,
    n_timesteps=50, convergence_thresh=10 * u.K, alpha=1, fluxes_up=None,
    fluxes_down=None
):
    """
    Compute emission spectrum.

    Parameters
    ----------
    opacities : dict
        Opacity database binned to wavelength grid.
    temperatures : ~astropy.units.Quantity
        Temperature grid
    pressures : ~astropy.units.Quantity
        Pressure grid
    lam : ~astropy.units.Quantity
        Wavelength grid
    F_TOA : ~astropy.units.Quantity
        Flux at the top of the atmosphere
    g : ~astropy.units.Quantity
        Surface graivty
    m_bar : ~astropy.units.Quantity
        Mean molecular weight
    n_timesteps : int
        Maximum number of timesteps in iteration for radiative equilibrium
    convergence_thresh : ~astropy.units.Quantity
        When the maximum change in temperature between timesteps is less than
        ``convergence_thresh``, accept this timestep as "converged".

    Returns
    -------
    F_2_up : ~astropy.units.Quantity
        Outgoing flux
    final_temps : ~astropy.units.Quantity
        Final temperature grid
    temperature_history : ~astropy.units.Quantity
        Grid of temperatures with dimensions (n_layers, n_timesteps)
    dtaus : ~numpy.ndarray
        Change in optical depth in final iteration
    """
    n_layers = len(pressures)
    n_wavelengths = len(lam)

    if fluxes_up is None: 
        fluxes_up = np.zeros((n_layers, n_wavelengths)) * flux_unit
        fluxes_up[0] = np.pi * BB(temperatures[0])(lam)

    if fluxes_down is None:
        fluxes_down = np.zeros((n_layers, n_wavelengths)) * flux_unit
        fluxes_down[-1] = F_TOA

    # from bottom of the atmosphere
    temperature_history = np.zeros((n_layers, n_timesteps + 1)) * u.K
    temperature_history[:, 0] = temperatures.copy()
    
    if n_timesteps > 1:
        timestep_iterator = trange(n_timesteps)

    else:
        timestep_iterator = np.arange(n_timesteps)

    for j in timestep_iterator:
        dtaus = [[1, ] * n_wavelengths]
        temps = temperature_history[:, j]
        temperature_changes = np.zeros(n_layers) * u.K
        
        for i in np.arange(0, n_layers - 1)[::-1]:
            p_2 = pressures[i + 1]
            T_2 = temps[i + 1]
            
            p_1 = pressures[i]
            T_1 = temps[i]

            k, sigma_scattering = kappa(
                opacities, T_1, p_1, lam, m_bar
            )

            delta_tau = delta_tau_i(
                k, p_1, p_2, g
            ).to(u.dimensionless_unscaled).value
            dtaus.append(delta_tau)
            # Single scattering albedo, Deitrick (2020) Eqn 17
            omega_0 = (
                sigma_scattering / (sigma_scattering + k)
            ).to(u.dimensionless_unscaled).value

            F_2_down = fluxes_down[i + 1]
            F_1_up = fluxes_up[i]
            
            F_2_up, F_1_down = propagate_fluxes(
                lam,
                F_1_up, F_2_down, T_1, T_2,
                delta_tau,
                omega_0=omega_0, g_0=0
            )
 
            fluxes_up[i + 1] = F_2_up
            fluxes_down[i] = F_1_down

            delta_F_i_dz, dz = div_bol_net_flux(
                bolometric_flux(F_2_up, lam), bolometric_flux(F_2_down, lam),
                bolometric_flux(F_1_up, lam), bolometric_flux(F_1_down, lam),
                T_1, T_2, p_1, p_2, g, alpha=alpha, m_bar=m_bar
            )

            dt = delta_t_i(p_1, p_2, T_1, T_2, delta_F_i_dz, g, m_bar=m_bar)
            temperature_changes[i] = delta_temperature(
                delta_F_i_dz, p_1, p_2, T_1, dt, g
            ).decompose()
            
        dT = u.Quantity(temperature_changes)
        temperature_history[:, j + 1] = temps - dT
        converged = np.all(np.abs(dT).max() < convergence_thresh)
        if n_timesteps > 1:
            timestep_iterator.set_description(
                f"max|∆T|={np.abs(dT).max():.1f}"
            )

            # Stop iterating if T-p profile changes by <convergence_thresh
            if converged:
                break

    return (
        fluxes_up, fluxes_down, temperature_history[:, j + 1], 
        temperature_history, np.array(dtaus), dT
    )
