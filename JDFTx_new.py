# Atomistic Simulation Environment (ASE) calculator interface for JDFTx
# See http://jdftx.org for JDFTx and https://wiki.fysik.dtu.dk/ase/ for ASE
# Authors: Deniz Gunceler, Ravishankar Sundararaman

from __future__ import print_function #For Python2 compatibility

import numpy as np
import scipy, subprocess, re
from ase.units import Bohr, Hartree
from ase import Atoms
from os import environ as env_vars_dict
from os.path import join as opj
from pymatgen.io.jdftx.outputs import JDFTXOutfile, JDFTXOutputs
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.jsonio import write_json
import ase.io
from ase.calculators.calculator import (
    Calculator,
    CalculatorSetupError,
    Parameters,
    all_changes
)
from pathlib import Path
from ase.config import cfg

#Run shell command and return output as a string:
def shell(cmd):
    return subprocess.check_output(cmd, shell=True)


def _get_atoms(infile: JDFTXInfile | None, atoms: Atoms | None) -> Atoms:
    if isinstance(atoms, Atoms):
        return atoms
    if isinstance(infile, JDFTXInfile):
        structure = infile.to_pmg_structure()
        atoms = AseAtomsAdaptor.get_atoms(structure)
        return atoms
    return None

def _tensor_to_voigt(tensor):
    return np.array(
        [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[1, 2], tensor[0, 2], tensor[0, 1]]
        )

def _strip_infile(infile: JDFTXInfile) -> JDFTXInfile:
    new_infile = infile.copy()
    strip_tags = ["ion", "lattice", "ion-species",]


class JDFTx(Calculator):
    
    implemented_properties = ['energy', 'forces', 'stress', 'charges']
    pseudoSetMap = {
        'SG15' : 'SG15/$ID_ONCV_PBE.upf',
        'GBRV' : 'GBRV/$ID_pbe.uspp',
        'GBRV_v1.5' : 'GBRV_v1.5/$ID_pbe_v1.uspp',
        'GBRV-pbe' : 'GBRV/$ID_pbe.uspp',
        'GBRV-lda' : 'GBRV/$ID_lda.uspp',
        'GBRV-pbesol' : 'GBRV/$ID_pbesol.uspp',
        'kjpaw': 'kjpaw/$ID_pbe-n-kjpaw_psl.1.0.0.upf',
        'dojo': 'dojo/$ID.upf',
        }
    pseudopotentials = ['fhi', 'uspp', 'upf', 'UPF', 'USPP']
    
    def __init__(
            self, restart=None, infile=None, pseudoDir=None, pseudoSet='GBRV',
            label='jdftx', atoms=None, command=None,
            debug=False, **kwargs
            ):
        atoms = _get_atoms(infile, atoms)
        self.infile = infile
        self.pseudoDir = pseudoDir
        self.pseudoSet = pseudoSet
        self._debug = debug
        self.label = None
        self.parameters = None
        self.results = None
        self.atoms = None
        self.command = command

        super().__init__(
            restart=restart,
            label=label, atoms=atoms, **kwargs
            )
        if restart is not None:
            self.read(restart)


          

    def _read_results(self, properties):
        outputs = JDFTXOutputs.from_calc_dir(self.directory, store_vars=["eigenvals"])
        outfile = outputs.outfile
        self.results["energy"] = outfile.e
        self.results["forces"] = outfile.forces
        self.results["charges"] = outfile.structure.site_properties["charges"]
        self.results["nbands"] = outfile.nbands
        self.results["nkpts"] = int(np.prod(outfile.kgrid))
        self.results["nspins"] = outfile.nspin
        self.results["fermi_level"] = outfile.efermi
        if not outfile.stress is None:
            self.results["stress"] = _tensor_to_voigt(outfile.stress)
        if not outfile.strain is None:
            self.results["strain"] = _tensor_to_voigt(outfile.strain)
        # Limit large output files to only those requested in properties
        if "eigenvalues" in properties:
            if not outputs.eigenvals is None:
                self.results["eigenvalues"] = outputs.eigenvals.copy().reshape(
                    (self.results["nspins"], self.results["nkpts"], self.results["nbands"])
                    )
        # TODO: Add occupations and kpoint_weights to JDFTxOutputs so I can add them here

    def write(self, label):
        self.atoms.write(label + "_restart.traj")
        #self.params.write(label + "_params.ase")
        with open(label + "_restart.json", "w") as f:
            ase.io.jsonio.write_json(f, self.results)

    def read(self, label):
        self.atoms = ase.io.read(label + "_restart.traj")
        #self.parameters = Parameters.read(label + "_params.ase")
        with open(label + "_restart.json", "r") as f:
            self.results = ase.io.jsonio.read_json(f)

    def _check_properties(self, properties):
        if "stress" in properties:
            self.infile.append_tag("dump", {"End": {"Stress": True}})

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self._check_properties(properties)
        self.constructInput(self.atoms)
        shell('cd %s && %s -i in -o out' % (self.directory, self.command))
        self._read_results(properties)
        self.write(self.label)
    
    def _get_ion_species_cmd(self, atomNames, pseudoDir):
        pseudoSetDir = opj(self.pseudoDir, self.pseudoSet)
        for_infile = {"ion-species": []}
        added = []  # List of pseudopotential that have already been added
        for atom in atomNames:
            if(sum([x == atom for x in added]) == 0.):
                for filetype in self.pseudopotentials:
                    try:
                        #print("trying", atom, filetype)
                        shell('ls %s | grep %s.%s' % (pseudoSetDir, atom, filetype))
                        for_infile["ion-species"].append('%s/%s.%s' % (pseudoSetDir, atom, filetype))
                        added.append(atom)
                        break
                    except Exception as e:
                        #print(e)
                        pass
                if not atom in added:
                    raise RuntimeError("Pseudopotential not found for atom %s in directory %s" % (atom, pseudoSetDir))
        return for_infile
    
    def constructInput(self, atoms: Atoms):
        structure = AseAtomsAdaptor.get_structure(atoms)
        if len(atoms.constraints):
            fixed_atom_inds = atoms.constraints[0].get_indices()
            structure.site_properties["selective_dynamics"] = [0 if i in fixed_atom_inds else 1 for i in range(len(atoms))]
        infile = JDFTXInfile.from_structure(structure)
        infile += self.infile
        if not "ion-species" in infile:
            infile += self._get_ion_species_cmd(atoms.get_chemical_symbols(), self.pseudoDir)
        infile.write_file(Path(self.directory) / "in")

