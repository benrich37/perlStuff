#!/usr/bin/env python

# Atomistic Simulation Environment (ASE) calculator interface for JDFTx
# See http://jdftx.org for JDFTx and https://wiki.fysik.dtu.dk/ase/ for ASE
# Authors: Deniz Gunceler, Ravishankar Sundararaman

from __future__ import print_function #For Python2 compatibility

import numpy as np
import scipy, subprocess, re
from ase.calculators.calculator import Calculator
# from ase.calculators.interface import Calculator
from ase.units import Bohr, Hartree
from ase import Atoms
from os import environ as env_vars_dict
from os.path import join as opj
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.inputs import JDFTXInfile


class JDFTx(Calculator):
    
    implemented_properties = ['energy', 'forces', 'stress', 'charges']

    def read(self, label):
        """
        Read the results from the JDFTx output file.
        """
        self.outfile = JDFTXOutfile.from_file(label + "out")

    
#     def __init__(self, infile: JDFTXInfile, command: str, **kwargs):
#         super().__init__(restart, ignore_bad_restart_file, label, atoms, directory, **kwargs)

