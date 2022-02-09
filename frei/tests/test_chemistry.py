import os
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import k_B

from ..opacity import iso_to_species
from ..chemistry import species_name_to_fastchem_name

test_pressures = np.logspace(-6, 2, 100) * u.bar
test_temperatures = 2400 * u.K * (test_pressures / (0.1 * u.bar)) ** 0.1

HAS_PYFASTCHEM = False

try:
    from pyfastchem import (
        FastChem, FastChemInput, FastChemOutput, FASTCHEM_UNKNOWN_SPECIES
    )
    HAS_PYFASTCHEM = True
except ImportError:
    from ..chemistry import (
        Mock_FastChemInput, mock_fastchem_output, Mock_FastChem
    )
    FastChem = Mock_FastChem
    FastChemInput = Mock_FastChemInput
    FastChemOutput = mock_fastchem_output(
        test_temperatures, test_pressures
    )
    FASTCHEM_UNKNOWN_SPECIES = 9999999

fastchem = FastChem(
    os.path.join(
        os.path.dirname(__file__), os.pardir, 'data',
        'element_abundances_solar.dat'
    ),
    os.path.join(
        os.path.dirname(__file__), os.pardir, 'data', 'logK.dat'
    ), 0
)


@pytest.mark.skipif(not HAS_PYFASTCHEM, reason='pyfastchem not installed')
@pytest.mark.parametrize("isotopologue, abund", (
    zip(['1H2-16O', 'Na', 'K', '48Ti-16O'], [3e-4, 3e-6, 1.8e-7, 1.4e-7])
),)
def test_max_abundances(isotopologue, abund):
    input_data = FastChemInput()
    output_data = FastChemOutput()

    input_data.temperature = test_temperatures.value[::-1]
    input_data.pressure = test_pressures.to(u.bar).value[::-1]

    fastchem.calcDensities(input_data, output_data)

    n_densities = np.array(output_data.number_densities) / u.cm ** 3

    gas_number_density = test_pressures[::-1] / (k_B * test_temperatures[::-1])

    species_name = iso_to_species(isotopologue)
    species_name_hill = species_name_to_fastchem_name(species_name)
    index = fastchem.getSpeciesIndex(species_name_hill)
    vmr = (
        n_densities[:, index] / gas_number_density
    ).to(u.dimensionless_unscaled).value
    np.testing.assert_allclose(vmr.max(), abund, rtol=0.1)


@pytest.mark.parametrize("isotopologue_name, species_name", (
    zip(['1H2-16O', 'Na', 'K', '48Ti-16O'], ["H2O", "Na", "K", "TiO"])
),)
def test_chemical_names_manipulation(isotopologue_name, species_name):
    # Test conversion of isotopologue name, like an opacity file with "1H2-16O"
    # to what I'll call the common "species name" like H2O
    assert iso_to_species(isotopologue_name) == species_name


@pytest.mark.parametrize("species_name, fastchem_name", (
    zip(['H2O', 'TiO', 'VO', 'Na', 'K', 'CO', 'CrH',
         'CF4O', 'Al2Cl6', 'AlNaF4', 'ClAlF2'],
        ['H2O1', 'O1Ti1', 'O1V1', 'Na', 'K', 'C1O1', 'Cr1H1',
         'C1F4O1', 'Al2Cl6', 'Al1F4Na1', 'Al1Cl1F2'])
),)
def test_chemical_names_manipulation(species_name, fastchem_name):
    # Test conversion of common species name to a fastchem name which can
    # be called in fastchem.getSpeciesIndex(fastchem_name)
    assert species_name_to_fastchem_name(species_name) == fastchem_name
