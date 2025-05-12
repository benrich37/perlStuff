# Atomistic Simulation Environment (ASE) calculator interface for JDFTx
# See http://jdftx.org for JDFTx and https://wiki.fysik.dtu.dk/ase/ for ASE
# Authors: Deniz Gunceler, Ravishankar Sundararaman
# Revised by Benjamin Rich

from __future__ import print_function #For Python2 compatibility

import numpy as np
import scipy, subprocess, re
from ase.units import Bohr, Hartree
from ase import Atoms
import warnings
from os import environ as env_vars_dict
from os.path import join as opj
from pymatgen.io.jdftx.outputs import JDFTXOutputs
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.jsonio import write_json
import ase.io
from ase.calculators.calculator import (
    Calculator,
    CalculatorSetupError,
    BaseCalculator,
    Parameters,
    all_changes
)
from pathlib import Path
from ase.config import cfg

run_dir_suffix = "jdftx_run"


class JDFTx(Calculator):

    """ Revised JDFTx calculator for ASE.

    Based on the original JDFTx calculator by Deniz Gunceler and Ravishankar Sundararaman.
    Rewritten for compatability with up-to-date ASE versions.  Requires pymatgen 2025.5.3 or later.

    Major Changes:
    - Uses JDFTx IO module in pymatgen to read and write input and output files.
        - Use of JDFTXInfile object 
    - Implements `read` and `write` methods to read and write restart files:
        - <label>_restart.traj
            - Trajectory file containing the atoms object. Overwrites the atoms object
              in the calculator with the most recent one (there is new functionality in
              ase to check for changes in an atoms object to decide if any properties
              need to be recalculated).
        - <label>_params.ase
            - ASE Parameters object containing all JDFTx input parameters non-redundant
              to the `atoms` object. This is pretty redundant at the moment to the JDFTXInfile
              object at the moment, but is slightly useful for restarting without needing to
              provide anything in `infile`.
        - <label>_restart.json
            - JSON file containing the results of the calculation, using the properties as stored in
              the `ase.Calculator` `results` dictionary field. This is where ASE expects to find any
              results from previous calculations.
        The motivation for for this implementation is primarily for NEB calculations, as this prevents
        redundant single-points to be run images not being updated between NEB steps.
    - Runs JDFTx in a sub-directory '<label>_jdftx_run', keeping the restart files separate from the state
      files. This is useful for ASE codes that require multiple sets of restart files to be present in the 
      same directory.
    
    """
    _debug = False
    default_parameters = {
        "dump-name": "$VAR",
        "initial-state": "$VAR",
        "dump": {"End": {"State": True}}
    }
    log_func = lambda _, x: print(x)
    
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
    pseudo_filetypes = ['fhi', 'uspp', 'upf', 'UPF', 'USPP']
    
    def __init__(
            self, 
            commands: dict | list | None = BaseCalculator._deprecated,
            executable: str | None = BaseCalculator._deprecated,
            restart: str | None = None, 
            infile: JDFTXInfile | dict | list | None = None, 
            pseudoDir=None, pseudoSet='GBRV',
            label='jdftx', atoms=None, command=None,
            detect_restart=True,
            ignore_state_on_failure=True,
            log_func = None,
            debug=True, **kwargs
            ):
        """ 

        JDFTx calculator for ASE.

        restart: str
            Restart label. Set if restarting from different label.
        infile: JDFTXInfile | dict | list | None
            JDFTx input file. If None, will use default parameters. This will probably break if you're
            not fetching parameters from a restart file.
        pseudoDir: str
            Directory containing pseudopotentials. If None, will use the JDFTx_pseudo environment variable.
        pseudoSet: str
            Pseudopotential set to use. Default is 'GBRV'. If None, will use the JDFTx_pseudo environment variable.
        label: str
            Label for the calculation. Default is 'jdftx'.
        atoms: Atoms
            Atoms object. If None, will use the atoms from the infile.
        command: str
            Command to run JDFTx.
        detect_restart: bool
            If True, will attempt a protected read of provided `label` for restart. Sets restart to `label` if
            successful.
        ignore_state_on_failure: bool
            If True and command returns an error, the input file will be rewritten with 'initial-state' removed and
            the calculation will be retried. If False, the calculation will fail. 
        debug: bool
            For debugging. Does nothing right now.
        """
        if not log_func is None:
            self.log_func = lambda x: log_func(x)
        commands = self._check_deprecated_keyword(commands, "commands")
        executable = self._check_deprecated_keyword(executable, "executable")
        if isinstance(label, str):
            _path = Path(label)
            if _path.exists():
                if _path.is_dir():
                    if not label.endswith("/"):
                        self.log_func(f"Label {label} is also directory - if you want to contain restart files in this directory, make sure your label ends with a '/'")
                    #     label += "/"
        self._set_infile(infile)
        atoms = _get_atoms(infile, atoms)
        self.pseudoDir = replaceVariable(pseudoDir, "JDFTx_pseudo")
        self.pseudoSet = pseudoSet
        self._debug = debug
        self.command = command
        self.ignore_state_on_failure = ignore_state_on_failure

        if (restart is None) and detect_restart:
            self.parameters = None
            try:
                self.read(label)
                restart = label
            except Exception as e:
                self.log_func(f"Restart detection failed: {e}")
                self.log_func("Continuing with new calculation.")

        super().__init__(
            restart=restart,
            label=label, atoms=atoms, **kwargs
            )
        if self._debug:
            self.log_func(f"Running {self.label} in debug mode")

    @property
    def run_dir(self):
        """Return the run directory for this calculation.

        Return the run directory for this calculation. Kept separate from the
        directory of the calculator to avoid confusion when restarting from
        state files.
        """
        prefix = self.prefix
        _run_dir = Path(self.directory) / f"{prefix}_{run_dir_suffix}"
        if prefix is None:
            _run_dir = Path(self.directory) / f"{run_dir_suffix}"
        if not _run_dir.exists():
            _run_dir.mkdir(parents=True)
        return str(_run_dir)
    
    def set_atoms(self, atoms):
        # This function is never called by JDFTx calculator, and is not even present
        # in the inherited Calculator class. However, it is needed for some reason, and
        # if the ".copy()" part is removed, atoms.calc.atoms == atoms, causing pretty much
        # every optimizer to break.
        self.atoms = atoms.copy()

    def _set_infile(self, infile):
        if self._debug:
            self.log_func(infile)
        infile_dict = self.default_parameters.copy()
        if isinstance(infile, JDFTXInfile):
            infile_dict.update(infile.as_dict())
        elif isinstance(infile, dict):
            infile_dict.update(infile)
        elif isinstance(infile, list):
            _infile_str = ""
            for v in infile:
                if isinstance(v, str):
                    _infile_str += v + "\n"
                elif isinstance(v, tuple):
                    _infile_str += " ".join(v) + "\n"
                elif isinstance(v, list):
                    _infile_str += " ".join([str(x) for x in v]) + "\n"
                else:
                    raise TypeError(f"Invalid type {type(v)} in infile list")
            self.log_func(_infile_str)
            _infile = JDFTXInfile.from_str(_infile_str, dont_require_structure=True)
            infile_dict.update(_infile.as_dict())
        elif isinstance(infile, str):
            _infile = JDFTXInfile.from_file(infile, dont_require_structure=True)
            infile_dict.update(_infile.as_dict())
        self.infile = _strip_infile(JDFTXInfile.from_dict(infile_dict))
        
          

    def _read_results(self, properties):
        outputs = JDFTXOutputs.from_calc_dir(self.run_dir, store_vars=["eigenvals"])
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
        self.parameters.write(label + "_params.ase")
        with open(label + "_restart.json", "w") as f:
            ase.io.jsonio.write_json(f, self.results)

    def read(self, label):
        self.atoms = ase.io.read(label + "_restart.traj")
        self.parameters = Parameters.read(label + "_params.ase")
        # parameters = Parameters.read(label + "_params.ase")
        # if (self.parameters is None) or (not len(self.parameters)):
        #     self.parameters = parameters
        # else:
        #     self.parameters.update(parameters)
        # if self.infile is None:
        #     self.infile = JDFTXInfile.from_dict(self.parameters)
        # else:
        #     self.parameters.update(self.infile.as_dict())
        with open(label + "_restart.json", "r") as f:
            self.results = ase.io.jsonio.read_json(f)

    # def _post_read(self):

    def _check_properties(self, properties):
        if "stress" in properties:
            self.infile.append_tag("dump", {"End": {"Stress": True}})

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if self._debug:
            self.log_func(f"Running in {self.run_dir}")
        Calculator.calculate(self, atoms, properties, system_changes)
        self._check_properties(properties)
        self.constructInput(atoms)
        self.run_jdftx()
        #shell('cd %s && %s -i in -o out' % (self.run_dir, self.command))
        self._read_results(properties)
        self.write(self.label)

    def run_jdftx(self, ran_before=False):
        try:
            shell('cd %s && %s -i in -o out' % (self.run_dir, self.command))
        except Exception as e:
            self.log_func(f"Error running JDFTx: {e}")
            if self.ignore_state_on_failure:
                if ran_before:
                    self.log_func("Ignoring state files did not work, aborting.")
                    raise RuntimeError("JDFTx calculation failed.")
                self.log_func("Ignoring state files and trying again.")
                self.constructInput(self.atoms, ignore_state=True)
                self.run_jdftx(ran_before=True)
            raise RuntimeError("JDFTx calculation failed.")

    
    def _get_ion_species_cmd(self, atomNames):
        pseudoSetDir = opj(self.pseudoDir, self.pseudoSet)
        for_infile = {"ion-species": []}
        added = []  # List of pseudopotential that have already been added
        for atom in atomNames:
            if(sum([x == atom for x in added]) == 0.):
                for filetype in self.pseudo_filetypes:
                    try:
                        shell('ls %s | grep %s.%s' % (pseudoSetDir, atom, filetype))
                        for_infile["ion-species"].append('%s/%s.%s' % (pseudoSetDir, atom, filetype))
                        added.append(atom)
                        break
                    except Exception as e:
                        pass
                if not atom in added:
                    raise RuntimeError("Pseudopotential not found for atom %s in directory %s" % (atom, pseudoSetDir))
        return for_infile
    
    def constructInput(self, atoms: Atoms | None, ignore_state=False):
        """Construct the input file for JDFTx.
        
        - Writes the 'in' file read by JDFTx `self.infile` and the provided `atoms`.
            - If `atoms` is None, will use `self.atoms`.
        - Sets self.parameters from the infile used to write the input file.
            - Ensures the most recent parameters are written upon `self.write()`.

        atoms: Atoms | None
            Atoms object to use for the calculation. If None, will use self.atoms.
        ignore_state: bool
            If True, will remove the 'initial-state' tag from the input file. This is useful for
            restarting calculations that have failed due to the initial state not being found.
        """
        if atoms is None:
            atoms = self.atoms
        structure = AseAtomsAdaptor.get_structure(atoms)
        if len(atoms.constraints):
            fixed_atom_inds = atoms.constraints[0].get_indices()
            structure.site_properties["selective_dynamics"] = [0 if i in fixed_atom_inds else 1 for i in range(len(atoms))]
        infile = JDFTXInfile.from_structure(structure)
        infile += self.infile
        if not "ion-species" in infile:
            infile += self._get_ion_species_cmd(atoms.get_chemical_symbols())
        if ignore_state and "initial-state" in infile:
            del infile["initial-state"]
        self.parameters = Parameters(_strip_infile(infile.copy()).as_dict())
        infile.write_file(Path(self.run_dir) / "in")

    def check_restart(self, label):
        """Returns a valid restart label.
        
        If the label is not valid, it will return None. Else, returns the provided label.
        Decides if label is valid by checking for errors upon running `self.read(label)`.
        """
        self.parameters = None
        try:
            self.read(label)
            return label
        except Exception as e:
            self.log_func(f"Restart detection failed: {e}")
            self.log_func("Continuing with new calculation.")
            return None

    def _check_deprecated_keyword(self, arg, argname):
        if arg is self._deprecated:
            return None
        else:
            warnings.warn(
                FutureWarning(
                    _keyword_deprecation_warnings[argname]
                )
            )
            return arg

    
#Run shell command and return output as a string:
def shell(cmd):
    return subprocess.check_output(cmd, shell=True)


def _get_atoms(infile: JDFTXInfile | None, atoms: Atoms | None) -> Atoms:
    if isinstance(atoms, Atoms):
        # Restarting a FIRE optimization for some reason retains atoms.calc.atoms == atoms
        # return atoms.copy()
        return atoms
    if isinstance(infile, JDFTXInfile):
        try:
            structure = infile.to_pmg_structure()
            atoms = AseAtomsAdaptor.get_atoms(structure)
            return atoms
        except:
            return None
    return None

def _tensor_to_voigt(tensor):
    return np.array(
        [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[1, 2], tensor[0, 2], tensor[0, 1]]
        )

def _strip_infile(infile: JDFTXInfile) -> JDFTXInfile:
    new_infile = infile.copy()
    strip_tags = ["ion", "lattice"]
    for tag in strip_tags:
        if tag in new_infile:
            del new_infile[tag]
    return new_infile


_commands_keyword_deprecation_warning = ''
'The keyword "commands" is deprecated and will be removed in future versions '
'if this revised calculator is adopted. Use "infile" instead, which can '
'accept a `JDFTXInfile` object, or a dictionary or list of strings.'

_executable_keyword_deprecation_warning = ''
'The keyword "executable" is deprecated and will be removed in future versions '
'if this revised calculator is adopted. Use "command" instead, which includes '
'passing the executable path to parallelization commands '
'(ie srun, mpirun, etc.). This most likely looks something like "srun -n 4 <jdftx path>" '


_keyword_deprecation_warnings = {
    "commands": _commands_keyword_deprecation_warning,
    "executable": _executable_keyword_deprecation_warning,
}


def replaceVariable(var, varName):
	if (var is None) and (varName in env_vars_dict):
		return env_vars_dict[varName]
	else:
		return var