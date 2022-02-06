import numpy as np
from astropy.constants import k_B, m_p, G, h, c
import astropy.units as u

__all__ = [
    'chemistry'
]


def chemistry(temperatures, pressures, input_data=None, output_data=None, return_vmr=False, m_bar=2.4*m_p):
    import pyfastchem
    
    if input_data is None and output_data is None: 
        fastchem = pyfastchem.FastChem(
            '/Users/brettmorris/git/FastChem/input/element_abundances_solar.dat', 
            '/Users/brettmorris/git/FastChem/input/logK.dat', 1
        )

        #create the input and output structures for FastChem
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()

    
    input_data.temperature = temperatures.value
    input_data.pressure = pressures.to(u.bar).value

    #run FastChem on the entire p-T structure
    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    n_densities = np.array(output_data.number_densities) / u.cm**3
    gas_number_density = pressures / (k_B * temperatures)
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
        if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            if return_vmr:
                fastchem_vmr[species_name] = (
                    n_densities[:, index] / gas_number_density
                ).to(u.dimensionless_unscaled).value
            fastchem_mmr[species_name] = (
                n_densities[:, index] / gas_number_density
            ).to(u.dimensionless_unscaled).value * (
                mass * m_p / m_bar
            ).to(u.dimensionless_unscaled).value
        else:
            print("Species", species, " not found in FastChem")

    if return_vmr: 
        return fastchem_mmr, fastchem_vmr
    return fastchem_mmr
