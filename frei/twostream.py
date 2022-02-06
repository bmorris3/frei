from tqdm import tqdm
import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_p, G, h, c, sigma_sb

from .opacity import kappa

__all__ = [
    'propagate_fluxes', 
    'emit'
]

flux_unit = u.erg / u.s / u.cm**3

def bolometric_flux(flux, lam): 
    return flux.to(flux_unit * u.cm, u.spectral_density(lam)).sum()

def delta_t_i(p_1, p_2, T_1, delta_F_i_dz, g, m_bar=2.4*m_p):
    dz = delta_z_i(T_1, p_1, p_2, g, m_bar)
    # Malik 2017 Eqn 28
    #f_i_pre = 1e5 / (abs(delta_F_i_dz * dz) / (u.erg/u.cm**2/u.s))**0.9
    f_i_pre = 2e5 / (abs(delta_F_i_dz * dz) / (u.erg/u.cm**2/u.s))**0.9
    # Malik 2017 Eqn 27
    return (f_i_pre * c_p() * p_1 / sigma_sb / g / T_1**3)

def BB(temperature):
    """
    Compute the blackbody flux as a function of wavelength `lam` 
    """
    # h = 6.62607015e-34  # J s
    # c = 299792458.0  # m/s
    # k_B = 1.380649e-23  # J/K

    return lambda wavelength: (
        2 * h * c**2 / np.power(wavelength, 5) /
        np.expm1(h * c / (wavelength * k_B * temperature))
    )

def E(omega_0, g_0): 
    # Deitrick (2020) Eqn 19
    return np.where(
        omega_0 > 0.1, 
        1.225 - 0.1582 * g_0 - 0.1777*omega_0 - 0.07465 * 
        g_0 ** 2 + 0.2351*omega_0*g_0 - 0.05582*omega_0**2,
        1
    )

def propagate_fluxes(lam, F_1_up, F_2_down, T_1, T_2, delta_tau, omega_0=0, g_0=0, eps=0.5):
    """
    Compute fluxes up and down using the improved two-stream equations.
    """
    # Malik 2017 Equation 6
    #T = (1 - delta_tau) * np.exp(-delta_tau) + delta_tau**2 * exp1(delta_tau)
    
    # Deitrick 2020 Equation B2
    T = np.exp(-2 * (E(omega_0, g_0) * (E(omega_0, g_0) - omega_0) * 
                     (1 - omega_0 * g_0))**0.5 * delta_tau)

    # Malik 2017 Equation 13
    zeta_plus = 0.5 * (1 + ((E(omega_0, g_0) - omega_0)/E(omega_0, g_0)/(1 - omega_0 * g_0))**0.5)
    zeta_minus = 0.5 * (1 - ((E(omega_0, g_0) - omega_0)/E(omega_0, g_0)/(1 - omega_0 * g_0))**0.5)

    # Malik 2017 Equation 12
    alpha = zeta_minus**2 * T**2 - zeta_plus**2
    beta = zeta_plus * zeta_minus * (1 - T**2)
    xi = (zeta_minus**2 - zeta_plus**2) * T
    nu = (zeta_minus**2 * T + zeta_plus**2) * (1 - T)
    pi_sr = np.pi
    # Malik 2017 Equation 5
    Bprime = lambda lam: (
        BB(T_2)(lam) - BB(T_1)(lam)
    ) / delta_tau 

    # Malik 2017 Equation 15, corrected from Dietrick 2022 Eqn B4
    F_2_up = (
        1/alpha * (
            xi * F_1_up - 
            beta * F_2_down + 
            2 * pi_sr * eps * (BB(T_1)(lam) * (alpha + beta) - 
                               BB(T_2)(lam)*xi + (eps / (1 - omega_0 * g_0)) * 
                               Bprime(lam) * (alpha - xi - beta))
        )
    )
    F_1_down = (
        1/alpha * (
            xi * F_2_down - 
            beta * F_1_up + 
            2 * pi_sr * eps * (BB(T_2)(lam) * (alpha + beta) - 
                               BB(T_1)(lam)*xi + (eps / (1 - omega_0 * g_0)) * 
                               Bprime(lam) * (xi - alpha + beta))
        )
    )
    return F_2_up, F_1_down

def delta_z_i(temperature_i, pressure_i, pressure_ip1, g, m_bar=2.4*m_p): 
    return (k_B * temperature_i) / (m_bar * g) * np.log(pressure_i / pressure_ip1)

def div_bol_net_flux(F_ip1_u, F_ip1_d, F_i_u, F_i_d, temperature_i, pressure_i, pressure_ip1, g, m_bar=2.4*m_p):
    # Malik 2017 Eqn 23
    return (
        (F_ip1_u - F_ip1_d) - (F_i_u - F_i_d)
    ) / delta_z_i(temperature_i, pressure_i, pressure_ip1, g, m_bar)

def delta_temperature(div, p_1, p_2, T_1, delta_t_i, g, m_bar=2.4 * m_p, n_dof=5):
    """
    Parameters
    ----------
    div : 
        Divergence of the bolometric net flux 
    rho_p : 
        local density
    delta_t_i : 
        numerical timestep
    """
    # Malik 2017 Eqn 24
    return 1/rho_p(p_1, p_2, T_1, g)/c_p(m_bar, n_dof) * div * delta_t_i
    
def c_p(m_bar=2.4 * m_p, n_dof=5): 
    # Malik 2017 Eqn 25
    return (2 + n_dof) / (2 * m_bar) * k_B

def delta_tau_i(kappa_i, p_1, p_2, g):
    return (p_1 - p_2) / g * kappa_i

def rho_p(p_1, p_2, T_1, g, m_bar=2.4*m_p):
    return ((p_1 - p_2) / g) / delta_z_i(T_1, p_1, p_2, g, m_bar)

def emit(opacities, temperatures, pressures, lam, F_TOA, g, m_bar=2.4*m_p, n_timesteps=50, plot=False):
    """
    Compute emission spectrum. 
    """
    fluxes_upwards = []
    n_layers = len(pressures)
    # from bottom of the atmosphere
    temperature_history = np.zeros((n_layers, n_timesteps)) * u.K
    temperature_history[:, 0] = temperatures.copy()

    dT_max = 100 * u.K * np.ones(n_layers)
    dT_min = - 100 * u.K * np.ones(n_layers)

    for j in tqdm(np.arange(n_timesteps - 1)):
        dtaus = []
        temps = temperature_history[:, j]
        temperature_changes = np.zeros(n_layers) * u.K

        for i in np.arange(n_layers):
            if i == 0: 
                # bottom of the atmosphere
                p_2 = pressures[i + 1]
                p_1 = pressures[i]
                T_2 = temps[i + 1]
                T_1 = temps[i]
                F_1_up = np.pi * BB(2 * T_1)(lam) # BOA
                F_2_down = np.pi * BB(T_2)(lam)

            elif i < len(pressures) - 1:
                # non-edge layers
                p_2 = pressures[i + 1]
                p_1 = pressures[i]
                T_2 = temps[i + 1]
                T_1 = temps[i]

                F_1_up = F_2_up
                F_2_down = np.pi * BB(T_2)(lam)

            else: 
                # Top of the atmosphere -- this isn't right
                p_2 = pressures[i] / (pressures[i-1] / pressures[i])
                p_1 = pressures[i]
                T_2 = temps[i] / (temps[i-1] / temps[i])
                T_1 = temps[i]

                F_1_up = F_2_up
                F_2_down = F_TOA
            k, sigma_scattering = kappa(
                opacities, T_1, p_1, lam, m_bar
            )

            # Single scattering albedo, Deitrick (2020) Eqn 17
            omega_0 = (
                sigma_scattering / (sigma_scattering + k)
            ).to(u.dimensionless_unscaled).value

            delta_tau = delta_tau_i(k, p_1, p_2, g).to(u.dimensionless_unscaled).value
            dtaus.append(delta_tau)

            F_2_up, F_1_down = propagate_fluxes(
                lam, 
                F_1_up, F_2_down, T_1, T_2, 
                delta_tau, 
                omega_0=omega_0, g_0=0
            )

            delta_F_i_dz = div_bol_net_flux(
                bolometric_flux(F_2_up, lam), bolometric_flux(F_2_down, lam), 
                bolometric_flux(F_1_up, lam), bolometric_flux(F_1_down, lam), 
                T_1, p_1, p_2, g
            )
            dt = delta_t_i(p_1, p_2, T_1, delta_F_i_dz, g)
            temperature_changes[i] = delta_temperature(
                delta_F_i_dz, p_1, p_2, T_1, dt, g
            ).decompose()

        temperature_history[:, j] = temps
        dT = u.Quantity(temperature_changes)
        temperature_history[:, j+1] = temps - dT 

        # Stop iterating if T-p profile changes by <10 K
        if np.abs(dT).max() < 10 * u.K:
            break

    if plot: 
        from astropy.visualize import quantity_support
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        with quantity_support():
            ax[0].loglog(lam, F_2_up.to(flux_unit), label=p_1)

            for i in range(n_timesteps):
                color = plt.cm.viridis(i / n_timesteps)
                ax[1].loglog(temperature_history[:, i], pressures, c=color);
            ax[1].semilogy(phoenix_temperatures, pressures, color='r', lw=3)
            ax[1].invert_yaxis();

        plt.legend(loc=(1, 0))
    return F_2_up, temps, temperature_history, dtaus