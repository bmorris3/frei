import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from ..core import Grid, Planet, effective_temperature
from ..opacity import load_example_opacity, kappa


def test_grid_init():
    planet = Planet.from_hot_jupiter()

    grid = Grid(
        planet=planet
    )

    for attr in ['lam', 'init_temperatures', 'pressures']:
        assert hasattr(grid, attr)


def test_example_opacities():
    planet = Planet.from_hot_jupiter()
    T_ref = 2400 * u.K

    grid = Grid(
        planet=planet,
        T_ref=T_ref
    )

    op = grid.load_opacities(opacities=load_example_opacity(grid))
    # Check that the synthetic opacities are loaded into the H2O key:
    assert "H2O" in op
    for attr in ['wavelength', 'temperature', 'pressure']:
        assert hasattr(op.get('H2O'), attr)

    k, sigma_scattering = kappa(
        op,
        grid.init_temperatures[0],
        grid.pressures[0],
        grid.lam,
        m_bar=planet.m_bar
    )
    # Synthetic example opacity is greater than scattering everywhere
    assert all(k > sigma_scattering)
    # Scattering cross section goes like Rayleigh, so should be decreasing
    assert sigma_scattering[0] > sigma_scattering[-1]

    spec, temps, temp_hist, dtaus = grid.emission_spectrum(n_timesteps=1)

    for attr in ['wavelength', 'flux']:
        assert hasattr(spec, attr)

    # Wavelength at peak flux should be close to this:
    assert_quantity_allclose(
        spec.wavelength[spec.flux.argmax()], 1.1257 * u.um, atol=0.01 * u.um
    )

    # Flux at peak flux should be close to this:
    flux_unit = u.erg/u.s/u.cm**3
    assert_quantity_allclose(
        spec.flux.max().to(flux_unit),
        1.255e13 * flux_unit,
        atol=0.01e13 * flux_unit
    )

    assert_quantity_allclose(
        effective_temperature(grid, spec, dtaus, temps),
        T_ref,
        atol=150 * u.K
    )
