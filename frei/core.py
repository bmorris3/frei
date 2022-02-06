import sys
sys.path.insert(0, '/Users/brettmorris/git/FastChem/python/')

from glob import glob
import gc
import numpy as np
import astropy.units as u
import logging
from dask.distributed import Client, LocalCluster

import astropy.units as u

def dask_client(memory_limit='20 GiB'):
    cluster = LocalCluster(
        memory_limit=memory_limit
    )
    client = Client(cluster)
    return client

def wavelength_grid(min_micron=0.5, max_micron=10, n_bins=500):
    lam = np.logspace(np.log10(min_micron), np.log10(max_micron), n_bins) * u.um
    wl_bins = np.concatenate([
        [(lam.min() - (lam[1] - lam[0])).to(u.um).value], 
        lam.to(u.um).value]
    ) + (lam[1] - lam[0]).to(u.um).value / 2
    R = float(lam[lam.shape[0]//2] / (lam[lam.shape[0]//2 + 1] - lam[lam.shape[0]//2]))
    return lam, wl_bins, R

def F_TOA(lam, f=2/3, a_rstar=float(0.03 * u.AU / u.R_sun)):
    return (f * a_rstar ** -2 * 
        1 / (2 * np.pi) * 
        (np.pi * u.sr * B_star(lam))
    ).to(u.erg/u.s/u.cm**3, u.spectral_density(lam))

