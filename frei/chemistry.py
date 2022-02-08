import os
import numpy as np
from astropy.constants import k_B, m_p, G, h, c
import astropy.units as u

__all__ = [
    'chemistry'
]


def chemistry(
        temperatures, pressures, return_vmr=False, m_bar=2.4*m_p
):
    """
    Run pyfastchem to compute chemistry throughout an atmosphere.

    Parameters
    ----------
    temperatures : ~astropy.units.Quantity
        Temperature grid
    pressures : ~astropy.units.Quantity
        Pressure grid
    fastchem : ~pyfastchem.FastChem
        FastChem object from pyfastchem
    input_data : ~pyfastchem.FastChemInput or None
    output_data : ~pyfastchem.FastChemOutput or None
    return_vmr : bool
        If True, return the volume mixing ratio as well as the mass mixing ratio
    m_bar : ~astropy.units.Quantity
        Mean molecular weight

    Returns
    -------
    fastchem_mmr : dict
        Mass mixing ratios for each species
    fastchem_vmr : dict (optional)
        Volume mixing ratios for each species
    """
    # If pyfastchem is not installed, mock it:
    try:
        from pyfastchem import (
            FastChem, FastChemInput, FastChemOutput, FASTCHEM_UNKNOWN_SPECIES
        )
    except ImportError:
        FastChem = Mock_FastChem
        FastChemInput = Mock_FastChemInput
        FastChemOutput = mock_fastchem_output(
            temperatures, pressures
        )
        FASTCHEM_UNKNOWN_SPECIES = 9999999

    fastchem = FastChem(
        os.path.join(
            os.path.dirname(__file__), 'data',
            'element_abundances_solar.dat'
        ),
        os.path.join(
            os.path.dirname(__file__), 'data', 'logK.dat'
        ), 1
    )
    fastchem.setVerboseLevel(0)

    #create the input and output structures for FastChem
    input_data = FastChemInput()
    output_data = FastChemOutput()
    
    input_data.temperature = temperatures.value[::-1]
    input_data.pressure = pressures.to(u.bar).value[::-1]

    #run FastChem on the entire p-T structure
    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    n_densities = np.array(output_data.number_densities) / u.cm**3

    gas_number_density = pressures[::-1] / (k_B * temperatures[::-1])
    # Hill notation, common spelling, mass
    all_species = [
        ['H2O1', 'H2O', 16+2], 
        ['O1Ti1', 'TiO', 16+48], 
        ['O1V1', 'VO', 16+51], 
        ['Na1+', 'Na', 23]
    ]

    if return_vmr:
        fastchem_vmr = dict()
    fastchem_mmr = dict()
    for i, (species_name_hill, species_name, mass) in enumerate(all_species):
        index = fastchem.getSpeciesIndex(species_name_hill)
        if index != FASTCHEM_UNKNOWN_SPECIES:
            vmr = (
                n_densities[:, index] / gas_number_density
            ).to(u.dimensionless_unscaled).value
            
            if len(vmr.shape) > 0:
                vmr = vmr[::-1]
            
            if return_vmr:
                fastchem_vmr[species_name] = vmr
            fastchem_mmr[species_name] = vmr * (
                mass * m_p / m_bar
            ).to(u.dimensionless_unscaled).value
        else:
            print("Species", species_name, "not found in FastChem")

    if return_vmr:
        return fastchem_mmr, fastchem_vmr
    return fastchem_mmr

# Mocking machinery for when pyfastchem is not installed (useful for tests)


class Mock_pyfastchem(object):
    def __init__(self):
        pass


class Mock_FastChem(object):
    def __init__(self, *args):
        pass

    def calcDensities(self, input, output):
        return 0

    def getSpeciesIndex(self, species_name):
        return 0


class Mock_FastChemInput(object):
    def __init__(self):
        self.temperature = None
        self.pressure = None


def mock_fastchem_output(temperatures, pressures):

    class Mock_FastChemOutput(object):
        def __init__(self):
            self.temperatures = temperatures[::-1]
            self.pressures = pressures[::-1]

        @property
        def number_densities(self):
            gas_number_density = self.pressures / (k_B * self.temperatures)
            return (
                1.5e-3 * gas_number_density[:, None]
            ).to(u.cm**-3).value

    return Mock_FastChemOutput