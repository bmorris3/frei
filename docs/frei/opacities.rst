Downloading opacities
=====================

Here's how to download opacities via DACE.

.. warning::
    These files are *huge*. Typical files are 5 GB per species. Make sure you
    have sufficient disk space before executing the code below!

The current behavior that is triggered is:

* a subdirectory named ``tmp/`` is created in the directory where the code is run

* binary opacity files will be unpacked into a temporary subdirectory within ``tmp/``

* the binary files will be read individually and concatenated into a netCDF file

* a netCDF file will be written to the user's home directory in a subdirectory
  called ``~/.frei``. If you ever need to clear up disk space, this is where to
  look

.. note::

    These functions require the DACE python client, available from pip via
    ``python -m pip install python-dace-client``

.. code-block:: python

    # Only run this code if you have sufficient disk space!
    from frei import download_molecule, download_atom

    # File download sizes: 4.8, 6.5 and 5.6 GB per file:
    isotopologues = ['48Ti-16O', '1H2-16O', '51V-16O']
    linelists = ['Toto', 'POKAZATEL', 'VOMYT']

    for isotopologue, linelist in zip(isotopologues, linelists):
        download_molecule(isotopologue, linelist)

    # File download size: 300 MB
    download_atom("Na", 0, "Kurucz")