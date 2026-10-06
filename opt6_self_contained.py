# For using pyjdftx with ASE optimizers

import os
from os.path import exists as ope, join as opj
from ase.io import read, write as _write
from ase.optimize import FIRE
from datetime import datetime
# from helpers.generic_helpers import optimizer, remove_dir_recursive, get_cmds_dict, get_apply_freeze_func
# from helpers.generic_helpers import get_log_fn, dump_template_input, read_pbc_val, is_head
# from helpers.generic_helpers import add_cohp_cmds, get_atoms_from_out, add_elec_density_dump
# from helpers.generic_helpers import log_def, check_structure, log_and_abort, cmds_dict_to_list, cmds_list_to_infile
from scripts.run_ddec6_v3 import main as run_ddec6
from sys import exit, stderr
from os import getcwd
from pymatgen.io.jdftx.inputs import JDFTXInfile
from JDFTx_pyjdftx import translate_infile_to_pydftx_kwargs, strip_infile_of_reserved_commands
from pathlib import Path

cwd = getcwd()
debug = "perlStuff" in cwd



##################################################################################################
##################################################################################################
############################START COPY PASTE OF GENERIC HELPERS###################################
##################################################################################################
##################################################################################################
from shutil import copy as cp
import numpy as np
from datetime import datetime as dt
from ase import Atoms, Atom
from ase.constraints import FixAtoms
from os.path import join as opj, exists as ope
from os import listdir as listdir, getcwd, chdir, listdir
from os import remove as rm, rmdir as rmdir, walk, getenv
from ase.io import read, write
from ase.units import Hartree
from pathlib import Path
from subprocess import run as run
import __main__
from pymatgen.io.jdftx.outputs import JDFTXOutfile, JDFTXOutfileSlice
from pymatgen.io.jdftx.joutstructures import JOutStructures, JOutStructure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.inputs import JDFTXInfile, clean_lines


def log_def(s):
    print(s)


state_files = ["wfns", "eigenvals", "fillings", "fluidState"]

gbrv_15_ref = [
    "sn f ca ta sc cd sb mg b se ga os ir li si co cr pt cu i pd br k as h mn cs rb ge bi ag fe tc hf ba ru al hg mo y re s tl te ti be p zn sr n rh au hf nb c w ni cl la in v pb zr o ",
    "14. 7. 10. 13. 11. 12. 15. 10. 3. 6. 19. 16. 15. 3. 4. 17. 14. 16. 19. 7. 16. 7. 9. 5. 1. 15. 9. 9. 14. 15. 19. 16. 15. 12. 10. 16. 3. 12. 14. 11. 15. 6. 13. 6. 12. 4. 5. 20. 10. 5. 15. 11. 12. 13. 4. 14. 18. 7. 11. 13. 13. 14. 12. 6. "
]

valence_electrons = {
    'h': 1, 'he': 2,
    'li': 1, 'be': 2, 'b': 3, 'c': 4, 'n': 5, 'o': 6, 'f': 7, 'ne': 8,
    'na': 1, 'mg': 2, 'al': 3, 'si': 4, 'p': 5, 's': 6, 'cl': 7, 'ar': 8,
    'k': 1, 'ca': 2, 'sc': 2, 'ti': 2, 'v': 2, 'cr': 1, 'mn': 2, 'fe': 2, 'co': 2, 'ni': 2, 'cu': 1, 'zn': 2,
    'ga': 3, 'ge': 4, 'as': 5, 'se': 6, 'br': 7, 'kr': 8,
    'rb': 1, 'sr': 2, 'y': 2, 'zr': 2, 'nb': 1, 'mo': 1, 'tc': 2, 'ru': 2, 'rh': 1, 'pd': 0, 'ag': 1, 'cd': 2,
    'in': 3, 'sn': 4, 'sb': 5, 'te': 6, 'i': 7, 'xe': 8,
    'cs': 1, 'ba': 2, 'la': 2, 'ce': 2, 'pr': 2, 'nd': 2, 'pm': 2, 'sm': 2, 'eu': 2, 'gd': 3, 'tb': 3, 'dy': 3,
    'ho': 3, 'er': 3, 'tm': 2, 'yb': 2, 'lu': 2, 'hf': 2, 'ta': 2, 'w': 2, 're': 2, 'os': 2, 'ir': 2, 'pt': 2,
    'au': 1, 'hg': 2, 'tl': 3, 'pb': 4, 'bi': 5, 'po': 6, 'at': 7, 'rn': 8,
}



foo_str = "fooooooooooooo"
bar_str = "barrrrrrrrrrrr"

submit_gpu_perl_ref = [
    "#!/bin/bash",
    f"#SBATCH -J {foo_str}",
    "#SBATCH --time=1:00:00",
    f"#SBATCH -o {foo_str}.out",
    f"#SBATCH -e {foo_str}.err",
    "#SBATCH -q regular_ss11",
    "#SBATCH -N 1",
    "#SBATCH -c 32",
    "#SBATCH --ntasks-per-node=4",
    "#SBATCH -C gpu",
    "#SBATCH --gpus-per-task=1",
    "#SBATCH --gpu-bind=none",
    "#SBATCH -A m4025_g\n",
    "export JDFTx_NUM_PROCS=1",
    "export SLURM_CPU_BIND=\"cores\"",
    "export JDFTX_MEMPOOL_SIZE=36000",
    "export MPICH_GPU_SUPPORT_ENABLED=1\n",
    f"python {bar_str} > {foo_str}.out",
]


submit_cpu_perl_ref = [
    "#!/bin/bash",
    f"#SBATCH -J {foo_str}",
    "#SBATCH --time=1:00:00",
    f"#SBATCH -o {foo_str}.out",
    f"#SBATCH -e {foo_str}.err",
    "#SBATCH -q regular",
    "#SBATCH -N 1",
    "#SBATCH --ntasks-per-node=4",
    "#SBATCH -C cpu",
    "#SBATCH -A m4025",
    "#SBATCH --hint=nomultithread\n",
    "# module use /global/cfs/cdirs/m4025/Software/Perlmutter/modules",
    "# module load jdftx/cpu\n"
    "export SLURM_CPU_BIND=\"cores\"",
    f"python {bar_str} > {foo_str}.out",
]

jdftx_calc_params = {
    "elec-ex-corr": "gga",
    "van-der-waals": "D3",
    "elec-n-bands": "*",
    "kpoint-folding": "*",
    "electronic-minimize": "nIterations 100 energyDiffThreshold  1e-07",
    "elec-smearing": "Fermi 0.001",
    "elec-initial-magnetization": "0 no",
    "spintype": "z-spin",
    "core-overlap-check": "none",
    "converge-empty-states": "yes",
    "symmetries": "none",
    "elec-cutoff": "25",
}

jdftx_solv_params = {
    "fluid": "LinearPCM",
    "pcm-variant": "CANDLE",
    "fluid-solvent": "H2O",
    "fluid-cation": "Na+ 0.5",
    "fluid-anion": "F- 0.5"
}

SHE_work_val_Ha = 4.66/Hartree



def read_inputs_dict_helper(work_dir, inputs_name="inputs"):
    inpfname = opj(work_dir, inputs_name)
    with open(Path(inpfname)) as file:
        string = file.read()
    lines: list[str] = list(clean_lines(string.splitlines()))
    lines = JDFTXInfile._gather_tags(lines)
    if ope(inputs_name):
        ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "initial-state",
                  "lattice-type", "opt", "max_steps", "fmax", 
                  "optimizer", "pseudos", "logfile", "restart", "econv", "safe-mode"]
        input_cmds = {"dump End": ""}
        for i, line in enumerate(lines):
            if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                skip = False
                for ig in ignore:
                    if ig in line:
                        skip = True
                if "#" in line:
                    skip = True
                if "ASE" in line:
                    break
                if not skip:
                    cmd = line[:line.index(" ")]
                    rest = line.rstrip("\n")[line.index(" ") + 1:]
                    if cmd not in ignore:
                        if not "dump " in cmd:
                            input_cmds[cmd] = rest
                        else:
                            freq = rest.split(" ")[0]
                            vars = " " + " ".join(rest.split(" ")[1:])
                            if freq == "End":
                                input_cmds["dump End"] += vars
                            else:
                                dump_cmd = f"dump {freq}"
                                if not dump_cmd in input_cmds:
                                    input_cmds[dump_cmd] = vars
                                else:
                                    input_cmds[dump_cmd] += vars
        return input_cmds
    else:
        return None



def read_inputs_dict(work_dir, pseudoSet="GBRV", ref_struct=None, bias=0.0, inputs_name="inputs"):
    input_cmds = read_inputs_dict_helper(work_dir, inputs_name=inputs_name)
    #print(input_cmds)
    if not input_cmds is None:
        nbandkey = "elec-n-bands"
        if ref_struct is None:
            ref_struct = opj(work_dir, "POSCAR")
        if nbandkey in input_cmds and "*" in input_cmds[nbandkey]:
            input_cmds[nbandkey] = str(get_nbands(ref_struct, pseudoSet=pseudoSet))
        kfoldkey = "kpoint-folding"
        if kfoldkey in input_cmds and "*" in input_cmds[kfoldkey]:
            input_cmds[kfoldkey] = str(get_kfolding(ref_struct))
        biaskey = "target-mu"
        if biaskey in input_cmds and "*" in input_cmds[biaskey]:
            if not bias is None:
                input_cmds[biaskey] = str(bias_to_mu(bias))
            else:
                del input_cmds[biaskey]
    return input_cmds



def get_zval(el, ps_set):
    ps_top_dir = getenv("JDFTx_pseudo")
    ps_dir = opj(ps_top_dir, ps_set)
    fs = listdir(ps_dir)
    file = None
    for f in fs:
        if "." in f:
            if f.split(".")[1].lower() == "upf":
                _el = f.split(".")[0].lower()
                if "_" in _el:
                    _el = _el.split("_")[0]
                if el.lower() == _el:
                    file = f
                    break
                else:
                    continue
    with open(opj(ps_dir, file), "r") as f:
        for line in f:
            if "z_valence" in line.lower():
                zval = int(float(line.split("=")[1].strip().strip('"').strip()))
            elif "z valence" in line.lower():
                zval = int(float(line.strip().split()[0]))
    return zval


def get_kfolding(poscar_fname, kpt_density=24):
    atoms = read(poscar_fname)
    lengths = [np.linalg.norm(atoms.cell[i]) for i in range(3)]
    kfold = [str(int(np.ceil(kpt_density/lengths[i]))) for i in range(3)]
    return " ".join(kfold)



def get_nbands(poscar_fname, pseudoSet="GBRV"):
    atoms = read(poscar_fname)
    count_dict = {}
    for a in atoms.get_chemical_symbols():
        if a.lower() not in count_dict.keys():
            count_dict[a.lower()] = 0
        count_dict[a.lower()] += 1
    nval = 0
    for a in count_dict.keys():
        val = get_zval(a, pseudoSet)
        count = count_dict[a]
        nval += int(val) * int(count)
    return max([int(nval / 2) + 10, int((nval / 2) * 1.2)])




def dup_cmds_list(infile):
    lattice_line = None
    infile_cmds = [("dump", "End State")]
    ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump-name", "initial-state",
              "coulomb-interaction", "coulomb-truncation-embed"]
    with open(infile) as f:
        for i, line in enumerate(f):
            if "lattice " in line:
                lattice_line = i
            if lattice_line is not None:
                if i > lattice_line + 3:
                    if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                        skip = False
                        for ig in ignore:
                            if ig in line:
                                skip = True
                            elif line[:4] == "ion ":
                                skip = True
                        if not skip:
                            cmd = line[:line.index(" ")]
                            rest = line.rstrip("\n")[line.index(" ") + 1:]
                            if cmd not in ignore:
                                if not cmd == "dump":
                                    infile_cmds.append((cmd, rest))
    return infile_cmds



def get_atom_str(atoms, index):
    return f"{atoms.get_chemical_symbols()[index]}({index + 1})"


def get_log_file_name(work, calc_type):
    fname = opj(work, calc_type + ".iolog")
    return fname

def is_head():
    rank = None
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank
    except Exception as e:
        pass
    if rank is None:
        return True
    elif rank == 0:
        return True
    return False

def get_log_fn(work, calc_type, print_bool, restart=False, parallel=False):
    if parallel:
        if is_head():
            return get_log_fn(work, calc_type, print_bool, restart=restart, parallel=False)
        else:
            return lambda s: None
    fname = get_log_file_name(work, calc_type)
    if not restart:
        if ope(fname):
            rm(fname)
    else:
        if ope(fname):
            log_generic("-------------------------- RESTARTING --------------------------", work, fname, print_bool)
    return lambda s: log_generic(s, work, fname, print_bool)


def log_generic(message, work, fname, print_bool):
    message = str(message)
    if "\n" not in message:
        message = message + "\n"
    prefix = dt.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    message = prefix + message
    log_fname = opj(work, fname)
    if not ope(log_fname):
        with open(log_fname, "w") as f:
            f.write(prefix + "Starting\n")
            f.close()
    with open(log_fname, "a") as f:
        f.write(message)
        f.close()
    if print_bool:
        print(message)

def bias_to_mu(bias_str, v0 = 4.66):
    if type(bias_str) is str:
        voltage = float(bias_str.rstrip("V")) + v0
    else:
        voltage = bias_str + v0
    mu = - voltage / Hartree
    return mu

def dump_default_inputs(work_dir, ref_struct, pseudoSet="GBRV", log_fn=log_def, pbc=None, bias=0.0):
    input_cmds = jdftx_calc_params
    if input_cmds["elec-n-bands"] == "*":
        nbands = str(get_nbands(ref_struct, pseudoSet=pseudoSet))
        log_fn(f"Default nbands for {ref_struct} set to {nbands}")
        input_cmds["elec-n-bands"] = nbands
    if input_cmds["kpoint-folding"] == "*":
        kfold = str(get_kfolding(ref_struct))
        log_fn(f"Default k-point folding for {ref_struct} set to {kfold}")
        input_cmds["kpoint-folding"] = kfold
    if "target-mu" in input_cmds:
        if input_cmds["target-mu"] == "*":
            if not bias is None:
                input_cmds["target-mu"] = str(bias_to_mu(bias))
            else:
                del input_cmds["target-mu"]
    inputs_str = ""
    for k in input_cmds:
        inputs_str += f"{k} {input_cmds[k]}\n"
    if False in pbc:
        log_fn("Non-bulk calculation - adding default solvation parameters")
        for k in jdftx_solv_params:
            inputs_str += f"{k} {jdftx_solv_params[k]}\n"
    with open(opj(work_dir, "inputs"), "w") as f:
        f.write(inputs_str)
    msg = "Default inputs dumped - check params before proceeding"
    log_and_abort(msg, log_fn)


def get_cmds_dict(work_dir, ref_struct=None, bias=0.0, log_fn=log_def, pbc=None, inputs_name="inputs"):
    chdir(work_dir)
    if not ope(opj(work_dir, inputs_name)):
        if ope(opj(work_dir, "in")):
            return dup_cmds_list(opj(work_dir, "in"))
        else:
            dump_default_inputs(work_dir, ref_struct, log_fn=log_fn, pbc=pbc, bias=bias)
            msg = "No inputs or in file found - dumping template inputs"
            log_and_abort(msg, log_fn)
    else:
        return read_inputs_dict(work_dir, ref_struct=ref_struct, bias=bias, inputs_name=inputs_name)
    



def remove_dir_recursive(path, log_fn=log_def):
    log_fn(f"Removing directory {path}")
    for root, dirs, files in walk(path, topdown=False):  # topdown=False makes the walk visit subdirectories first
        for name in files:
            try:
                rm(opj(root, name))
            except FileNotFoundError as e:
                log_fn(f"File {opj(root, name)} not found - ignoring")
                log_fn(e)
                pass
        for name in dirs:
            rmdir(opj(root, name))
    rmdir(path)  # remove the root directory itself


def add_constraint(atoms, constraint, log_fn=log_def):
    consts = atoms.constraints
    if len(consts) == 0:
        atoms.set_constraint(constraint)
    else:
        consts.append(constraint)
        atoms.set_constraint(consts)


def get_freeze_surf_base_constraint_by_dist(atoms, ztol = 3., log_fn=log_def):
    direct_posns = np.dot(atoms.positions, np.linalg.inv(atoms.cell))
    for i in range(3):
        direct_posns[:, i] *= np.linalg.norm(atoms.cell[i])
    min_z = min(direct_posns[:, 2])
    mask = (direct_posns[:, 2] < (min_z + ztol))
    log_fn(f"Imposing atom freezing for atoms in bottom {ztol:1.1g}A")
    log_str = ""
    for i, m in enumerate(mask):
        if m:
            log_str += f"{get_atom_str(atoms, i)}, "
    log_fn(f"freezing {log_str}")
    c = FixAtoms(mask=mask)
    return c


def get_freeze_all_but_by_map(atoms, freeze_map: dict[str, list[int]], log_fn=log_def):
    return _get_freeze_by_map(atoms, freeze_map, True, log_fn=log_fn)

def get_freeze_by_map(atoms, freeze_map: dict[str, list[int]], log_fn=log_def):
    return _get_freeze_by_map(atoms, freeze_map, False, log_fn=log_fn)

def _get_freeze_by_map(atoms, freeze_map: dict[str, list[int]], ref_bool: bool, log_fn=log_def):
    mask = [ref_bool for _ in range(len(atoms))]
    for el in freeze_map:
        el_idcs = [idx for idx, _el in enumerate(atoms.get_chemical_symbols()) if el == _el]
        read_idcs = [i % len(el_idcs) for i in freeze_map[el]] # allow negative indexing
        log_fn(f"Map for freezing {el}: {freeze_map[el]} -> {read_idcs} of {el_idcs}")
        mapped_idcs = [idx for i, idx in enumerate(el_idcs) if i in read_idcs]
        log_fn(f"Mapped idcs for freezing {el}: {mapped_idcs}")
        for idx in mapped_idcs:
            mask[idx] = not mask[idx]
    log_fn(f"Imposing atom freezing by map")
    log_str = ""
    for i, m in enumerate(mask):
        if m:
            log_str += f"{get_atom_str(atoms, i)}, "
    log_fn(f"freezing {log_str}")
    c = FixAtoms(mask=mask)
    return c

def get_freeze_surf_base_constraint_by_idcs(atoms, freeze_idcs, log_fn=log_def):
    mask = []
    for i in range(len(atoms)):
        if i in freeze_idcs:
            mask.append(True)
        else:
            mask.append(False)
    log_fn(f"Imposing atom freezing for bottom {len(freeze_idcs)} atoms")
    log_str = ""
    for i, m in enumerate(mask):
        if m:
            log_str += f"{get_atom_str(atoms, i)}, "
    log_fn(f"freezing {log_str}")
    c = FixAtoms(mask=mask)
    return c

def get_freeze_surf_base_constraint_by_count(atoms: Atoms, freeze_count=1, exclude_freeze_count=0, log_fn=log_def):
    posns = atoms.get_positions()
    idcs = np.argsort(posns[:, 2])
    mask = []
    for a in atoms:
        mask.append(False)
    for i in range(exclude_freeze_count, freeze_count+exclude_freeze_count):
        mask[idcs[i]] = True
    log_fn(f"Imposing atom freezing for bottom {freeze_count} atoms")
    log_str = ""
    for i, m in enumerate(mask):
        if m:
            log_str += f"{get_atom_str(atoms, i)}, "
    log_fn(f"freezing {log_str}")
    c = FixAtoms(mask=mask)
    return c


def get_freeze_surf_base_constraint(atoms, ztol = 3., freeze_count = 0, exclude_freeze_count=0, freeze_idcs=None, freeze_map: dict | None = None, freeze_all_but_map: dict | None = None, log_fn=log_def):
    if freeze_idcs is None:
        freeze_idcs = []
    if freeze_map is None:
        freeze_map = {}
    if freeze_all_but_map is None:
        freeze_all_but_map = {}
    if len(freeze_all_but_map) > 0:
        return get_freeze_all_but_by_map(atoms, freeze_map=freeze_all_but_map, log_fn=log_fn)
    elif len(freeze_map) > 0:
        return get_freeze_by_map(atoms, freeze_map=freeze_map, log_fn=log_fn)
    elif len(freeze_idcs) > 0:
        return get_freeze_surf_base_constraint_by_idcs(atoms, freeze_idcs=freeze_idcs, log_fn=log_fn)
    elif freeze_count > 0:
        return get_freeze_surf_base_constraint_by_count(atoms, freeze_count=freeze_count, exclude_freeze_count=exclude_freeze_count, log_fn=log_fn)
    else:
        return get_freeze_surf_base_constraint_by_dist(atoms, ztol = ztol, log_fn=log_fn)
    
def get_apply_freeze_func(freeze_base, freeze_tol, freeze_count, freeze_idcs, exclude_freeze_count, freeze_map: dict | None = None, freeze_all_but_map: dict | None = None, log_fn=log_def):
    if freeze_idcs is None:
        freeze_idcs = []
    if freeze_map is None:
        freeze_map = {}
    if freeze_all_but_map is None:
        freeze_all_but_map = {}
    def apply_freeze_func(atoms, log_fn=log_fn):
        if any([freeze_base, bool(len(freeze_idcs)), bool(len(freeze_map)), bool(len(freeze_all_but_map))]):
            c = get_freeze_surf_base_constraint(
                atoms,
                ztol=freeze_tol, freeze_count=freeze_count, freeze_idcs=freeze_idcs, exclude_freeze_count=exclude_freeze_count,
                freeze_map=freeze_map, freeze_all_but_map=freeze_all_but_map,
                log_fn=log_fn)
            add_constraint(atoms, c)
            return atoms
        else:
            return atoms
    return apply_freeze_func



def optimizer(atoms, root, opter, opt_alpha=150, **kwargs):
    if not opt_alpha is None:
        kwargs["a"] = (opt_alpha / 70) * 0.1
    kwargs.pop("opt_alpha", None)
    traj = opj(root, "opt.traj")
    log = opj(root, "opt.log")
    restart = opj(root, "hessian.pckl")
    kwargs.update({"restart": restart})
    if is_head():
        kwargs.update({"trajectory": traj, "logfile": log})
    dyn = opter(atoms, **kwargs)
    return dyn


def get_inputs_list(fname, auto_lower=True):
    inputs = []
    with open(fname, "r") as f:
        for line in f:
            if ":" in line:
                key = line.split(":")[0]
                val = line.rstrip("\n").split(":")[1]
                if "#" in val:
                    val = val[:val.index("#")]
                if auto_lower:
                    key = key.lower()
                    val = val.lower()
                if "#" not in key:
                    inputs.append(tuple([key, val]))
    return inputs


submit_fname = "psubmit.sh"



def read_pbc_val(val):
    splitter = " "
    vsplit = val.strip().split(splitter)
    pbc = []
    for i in range(3):
        pbc.append("true" in vsplit[i].lower())
    return pbc





def append_key_val_to_cmds_list(cmds, key, val, allow_duplicates = False, append_duplicates = False):
    keys = [cmd[0] for cmd in cmds]
    if not key in keys:
        cmds.append((key, val))
    elif allow_duplicates:
        if append_duplicates:
            cmds[keys.index(key)][1] += f" {val}"
        else:
            cmds.append((key, val))
    else:
        cmds[keys.index(key)] = (key, val)
    return cmds


def add_elec_density_dump(cmds_list):
    key = "dump"
    val = "End ElecDensity"
    cmds_list = append_key_val_to_cmds_list(cmds_list, key, val, allow_duplicates=True)
    return cmds_list

has_subshells = {
    1: "s",
    3: "p",
    11: "d"
}

m_orbs_dict = {1: ['s'], 3: ['p','px','py','pz'], 19: ['d','dxy','dxz','dyz','dz2','dx2-y2']}




def cmds_dict_to_list(cmds_dict):
    cmds_list = []
    for k in cmds_dict:
        cmds_list.append([k, cmds_dict[k]])
    return cmds_list




def add_cohp_cmds(cmds, ortho=True):
    is_dict = (type(cmds) == dict)
    dump_ends = [
        "BandProjections", "Fillings", "Kpoints", "BandEigs"
    ]
    rest_pairs = [
        ["band-projection-params"]
    ]
    if ortho:
        rest_pairs[0].append("yes no")
    else:
        rest_pairs[0].append("no no")
    for dp in dump_ends:
        key = "dump End"
        val = dp
        cmds = append_key_val_to_cmds_list(cmds, key, val, allow_duplicates=True, append_duplicates=True)
    for rp in rest_pairs:
        key = rp[0]
        val = rp[1]
        cmds = append_key_val_to_cmds_list(cmds, key, val, allow_duplicates=False)
    return cmds



def log_and_abort(err_str, log_fn=log_def):
    if err_str[-1:] != "\n":
        err_str += "\n"
    log_fn(err_str)
    raise ValueError(err_str)


def check_structure(structure, work, log_fn=log_def):
    use_fmt = "vasp"
    fname_out = "POSCAR"
    suffixes = ["com", "gjf"]
    have_gauss = False
    if not ope(opj(work, structure)):
        log_fn(f"Could not find {structure} - checking if gaussian input")
        for s in suffixes:
            gauss_struct = structure + "." + s
            fpath = opj(work, gauss_struct)
            if ope(fpath):
                log_fn(f"Found matching gaussian input ({gauss_struct})")
                have_gauss = True
        if not have_gauss:
            err_str = f"Could not find {structure} - aborting"
            log_fn(err_str)
            raise ValueError(err_str)
        else:
            structure = gauss_struct
            use_fmt = "gaussian-in"
    elif "." in structure:
        log_fn(f"Checking if gave gaussian structure")
        suffix = structure.split(".")[1]
        if suffix in suffixes:
            use_fmt = "gaussian-in"
        else:
            log_fn(f"Not sure which format {structure} is in - setting format for reader to None")
            use_fmt = None
    structure = opj(work, structure)
    try:
        atoms_obj = read(structure, format=use_fmt)
    except Exception as e:
        log_fn(e)
        if isinstance(e, AttributeError) and use_fmt == "gaussian-in":
            log_fn("AttributeError - trying to clear unknown symbols from gaussian input")
            clear_unknown_symbols_from_gaussian_in(structure)
            atoms_obj = read(structure, format=use_fmt)
        else:
            log_fn(f"Could not read structure {structure} with format {use_fmt} - aborting")
            raise e
    log_fn(f"Saving found structure {structure} as {opj(work, fname_out)}")
    structure = opj(work, fname_out)
    write(structure, atoms_obj, format="vasp")
    return structure

def clear_unknown_symbols_from_gaussian_in(fname):
    with open(fname, "r") as f:
        _lines = f.readlines()
    lines = [line for line in _lines if not "?" in line]
    with open(fname, "w") as f:
        f.writelines(lines)



def get_atoms_from_pmg_joutstructure(jstruc: JOutStructure):
    struc: Structure = jstruc.structure
    atoms: Atoms = AseAtomsAdaptor.get_atoms(struc)
    E = 0
    if not jstruc.e is None:
        E = jstruc.e
    atoms.E = E
    charges = np.zeros(len(atoms))
    if not jstruc.charges is None:
        for i, charge in enumerate(jstruc.charges):
            charges[i] = charge
    atoms.set_initial_charges(charges)
    if "velocities" in struc.site_properties and struc.site_properties["velocities"] is not None:
        try:
            atoms.set_velocities(struc.site_properties["velocities"])
        except Exception as e:
            print(f"Error setting velocities for {atoms}: {e}")
    if hasattr(jstruc, "thermostat_velocity") and jstruc.thermostat_velocity is not None:
        try:
            atoms.info["thermostat-velocity"] = jstruc.thermostat_velocity
        except Exception as e:
            print(f"Error setting thermostat-velocity for {atoms}: {e}")
    return atoms

def get_atoms_list_from_pmg_joutstructures(jstrucs: JOutStructures):
    atoms_list = []
    for jstruc in jstrucs:
        atoms = get_atoms_from_pmg_joutstructure(jstruc)
        atoms_list.append(atoms)
    return atoms_list

def get_atoms_list_from_pmg_jdftxoutfileslice(jdftxoutfile_slice: JDFTXOutfileSlice):
    jstrucs = jdftxoutfile_slice.jstrucs
    return get_atoms_list_from_pmg_joutstructures(jstrucs)

def get_atoms_list_from_pmg_jdftxoutfile(jdftxoutfile):
    atoms_list = []
    for jdftxoutfile_slice in jdftxoutfile:
        if jdftxoutfile_slice is not None:
            atoms_list += get_atoms_list_from_pmg_jdftxoutfileslice(jdftxoutfile_slice)
        else:
            atoms_list.append(None)
    return atoms_list

def get_atoms_from_out(outfile_path, ):
    outfile = JDFTXOutfile.from_file(outfile_path, none_slice_on_error=True)
    atoms_list = get_atoms_list_from_out_alt(outfile)
    atoms_1 = atoms_list[-1]
    atoms_2 = None
    if len(atoms_list) > 1:
        atoms_2 = atoms_list[-2]
    if atoms_2 is None:
        return atoms_1
    else:
        # If calculation is AIMD, the last structure is likely partially written and
        # may be missing the thermostat-velocity. In that case, return the second to last structure.
        if outfile.slices[-1].is_md:
            return atoms_2
        else:
            return atoms_1


def get_atoms_list_from_out_alt(outfile):
    _atoms_list = get_atoms_list_from_pmg_jdftxoutfile(outfile)
    atoms_list = [a for a in _atoms_list if a is not None]
    if not len(atoms_list):
        atoms_list = [AseAtomsAdaptor.get_atoms(outfile.structure)]
    return atoms_list

    
def cmds_list_to_infile(cmds_list):
    infile_dict = {}
    if isinstance(cmds_list, JDFTXInfile):
        return cmds_list
    elif isinstance(cmds_list, dict):
        infile_dict.update(cmds_list)
    elif isinstance(cmds_list, list):
        _infile_str = ""
        for v in cmds_list:
            if isinstance(v, str):
                _infile_str += v + "\n"
            elif isinstance(v, tuple):
                _infile_str += " ".join(v) + "\n"
            elif isinstance(v, list):
                _infile_str += " ".join([str(x) for x in v]) + "\n"
            else:
                raise TypeError(f"Invalid type {type(v)} in infile list")
        _infile = JDFTXInfile.from_str(_infile_str, dont_require_structure=True)
        infile_dict.update(_infile.as_dict())
    elif isinstance(cmds_list, str):
        _infile = JDFTXInfile.from_file(cmds_list, dont_require_structure=True)
        infile_dict.update(_infile.as_dict())
    return JDFTXInfile.from_dict(infile_dict)
##################################################################################################
##################################################################################################
############################END COPY PASTE OF GENERIC HELPERS###################################
##################################################################################################
##################################################################################################

def write(fname, _atoms, format="vasp"):
    atoms = _atoms.copy()
    atoms.pbc = [True,True,True]
    _write(fname, atoms, format=format)





def read_opt_inputs(fname = "opt_input"):
    inputs = get_inputs_list(fname, auto_lower=False)
    opt_inputs_dict = {
        "structure": None,
        "fmax": 0.05,
        "max_steps": 100,
        "gpu": True,
        "restart": False,
        "pbc": None,
        "lat_iters": 0,
        "freeze_base": False,
        "freeze_tol": 3.,
        "ortho": True,
        "save_state": False,
        "pseudoSet": "GBRV",
        "bias": 0.0,
        "ddec6": True,
        "freeze_count": 0,
        "exclude_freeze_count": 0,
        "direct_coords": False,
        "freeze_map": None,
        "freeze_all_but_map": None,
    }
    for input in inputs:
        key, val = input[0], input[1]
        if "pseudo" in key:
            opt_inputs_dict["pseudoSet"] = val.strip()
        if "structure" in key:
            opt_inputs_dict["structure"] = val.strip()
        if "work" in key:
            opt_inputs_dict["work_dir"] = val
        if "gpu" in key:
            opt_inputs_dict["gpu"] = "true" in val.lower()
        if "restart" in key:
            opt_inputs_dict["restart"] = "true" in val.lower()
        if "max" in key:
            if "fmax" in key:
                opt_inputs_dict["fmax"] = float(val)
            elif "step" in key:
                opt_inputs_dict["max_steps"] = int(val)
        if "pbc" in key:
            opt_inputs_dict["pbc"] = read_pbc_val(val)
        if "lat" in key:
            try:
                opt_inputs_dict["n_iters"] = int(val)
                opt_inputs_dict["lat_iters"] = opt_inputs_dict["n_iters"]
            except:
                pass
        if ("direct" in key) and ("coord" in key):
            opt_inputs_dict["direct_coords"] = "true" in val.lower()
        if ("opt" in key) and ("progr" in key):
            if "jdft" in val:
                opt_inputs_dict["use_jdft"] = True
            if "ase" in val:
                opt_inputs_dict["use_jdft"] = False
            else:
                pass
        if ("freeze" in key):
            if ("base" in key):
                opt_inputs_dict["freeze_base"] = "true" in val.lower()
            elif ("tol" in key):
                opt_inputs_dict["freeze_tol"] = float(val)
            elif ("count" in key):
                if "exclude" in key:
                    opt_inputs_dict["exclude_freeze_count"] = int(val)
                else:
                    opt_inputs_dict["freeze_count"] = int(val)
            elif ("map" in key):
                freeze_dict = parse_dict_indexing(val)
                if "all_but" in key:
                    opt_inputs_dict["freeze_all_but_map"] = freeze_dict
                else:
                    opt_inputs_dict["freeze_map"] = freeze_dict
        if ("ortho" in key):
            opt_inputs_dict["ortho"] = "true" in val.lower()
        if ("save" in key) and ("state" in key):
            opt_inputs_dict["save_state"] = "true" in val.lower()
        if "bias" in key:
            v= val.strip()
            if "V" in v:
                v = v.rstrip("V")
            if "none" in v.lower() or "no" in v.lower():
                opt_inputs_dict["bias"] = None
            else:
                try:
                    opt_inputs_dict["bias"] = float(v)
                except Exception as e:
                    print(e)
                    print("Assigning no bias")
                    opt_inputs_dict["bias"] = None
        if "ddec6" in key:
            opt_inputs_dict["ddec6"] = "true" in val.lower()
    opt_inputs_dict["work_dir"] = getcwd()
    return opt_inputs_dict
    # return work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters, use_jdft, freeze_base, freeze_tol, ortho, save_state, pseudoset, bias, ddec6, freeze_count

def parse_dict_indexing(line: str):
    # Expected format: (el1, i1, i2, ...), (el2, i1, i2, ...), ...
    pieces = []
    start_idcs = [i for i, v in enumerate(line) if v == "("]
    end_idcs = [i for i, v in enumerate(line) if v == ")"]
    for start, end in zip(start_idcs, end_idcs):
        pieces.append(line[start + 1:end].strip())
    index_dict = {}
    for piece in pieces:
        el, *indices = [v.strip() for v in piece.split(',')]
        index_dict[el.strip()] = [int(i.strip()) for i in indices]
    return index_dict

def finished(dirname):
    with open(opj(dirname, "finished.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")


def get_atoms_from_lat_dir(dir):
    outfile = opj(dir, "out")
    return get_atoms_from_out(outfile)

def make_dir(dirname):
    if not ope(dirname):
        os.mkdir(dirname)



def get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_def, prefix="jdftx."):
    atoms_obj = None
    outfile_path1 = opj(opt_dir, f"{prefix}out")
    outfile_path2 = opj(opt_dir, opj("jdftx_run", f"{prefix}out"))
    if ope(outfile_path2):
        try:
            atoms_obj = get_atoms_from_out(outfile_path2)
            log_fn(f"Atoms object set from {outfile_path2}")
        except Exception as e:
            log_fn(f"Error reading atoms object from {outfile_path2}")
            pass
    if atoms_obj is None:
        if ope(outfile_path1):
            try:
                atoms_obj = get_atoms_from_out(outfile_path1)
                log_fn(f"Atoms object set from {outfile_path1}")
            except Exception as e:
                log_fn(f"Error reading atoms object from {outfile_path1}")
                pass
    return atoms_obj


def get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
    for path in [opt_dir, lat_dir]:
        if not ope(path):
            make_dir(path)
    # If an atoms is found in the ion_opt dir, then an ionic minimization was run
    # (So even if the lattice opt was run, the ion opt will be the most recent)
    atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn)
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn, prefix="")
    if not atoms is None:
        structure = opj(opt_dir, "POSCAR")
        write(structure, atoms, format="vasp")
        return structure, restart
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(lat_dir, log_fn=log_fn)
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(lat_dir, log_fn=log_fn, prefix="")
    if not atoms is None:
        structure = opj(lat_dir, "POSCAR")
        write(structure, atoms, format="vasp")
        return structure, restart
    if atoms is None:
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    return structure, restart


def get_restart_atoms(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
    # for path in [opt_dir, lat_dir]:
    #     if not ope(path):
    #         make_dir(path)
    # If an atoms is found in the ion_opt dir, then an ionic minimization was run
    # (So even if the lattice opt was run, the ion opt will be the most recent)
    atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn)
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn, prefix="")
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(lat_dir, log_fn=log_fn)
    if atoms is None:
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
            atoms = read(structure, format="vasp")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    return atoms, restart



def get_structure(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=log_def):
    dirs_list = [opt_dir]
    if lat_iters > 0:
        log_fn(f"Lattice opt requested ({lat_iters} iterations) - adding lat dir to setup list")
        dirs_list.append(lat_dir)
        dirs_list.append(opj(lat_dir, "jdftx_run"))
    if not restart:
        for d in dirs_list:
            if ope(d):
                log_fn(f"Resetting {d}")
                remove_dir_recursive(d)
            make_dir(d)
        if ope(structure):
            log_fn(f"Found {structure} for structure")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    else:
        structure, restart = get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_fn)
    return structure, restart


def get_atoms(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=log_def):
    dirs_list = [opt_dir]
    if lat_iters > 0:
        log_fn(f"Lattice opt requested ({lat_iters} iterations) - adding lat dir to setup list")
        dirs_list.append(lat_dir)
        dirs_list.append(opj(lat_dir, "jdftx_run"))
    if not restart:
        for d in dirs_list:
            if ope(d):
                log_fn(f"Resetting {d}")
                remove_dir_recursive(d)
            make_dir(d)
        if ope(structure):
            log_fn(f"Found {structure} for structure")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    else:
        atoms, restart = get_restart_atoms(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_fn)
    return atoms, restart







def get_pbc_from_infile(infile: JDFTXInfile):
    coulomb_intrx: dict = infile.get("coulomb-interaction", None)
    if coulomb_intrx is None:
        return None
    ttype = coulomb_intrx.get("truncationType")
    if ttype == "Periodic":
        return (True, True, True)
    elif ttype == "Isolated":
        return (False, False, False)
    elif ttype == "Slab":
        tdir = coulomb_intrx.get("dir", "001")
        return ((not bool(int(tdir[0]))), (not bool(int(tdir[1]))), (not bool(int(tdir[2]))))
    elif ttype == "Wire":
        tdir = coulomb_intrx.get("dir", "001")
        return ((not bool(int(tdir[0]))), (not bool(int(tdir[1]))), bool(int(tdir[2])))
    else:
        raise ValueError(f"Unrecognized coulomb truncation type {ttype} in infile.")
    
    


def check_pbc(pbc: list[bool] | tuple[bool, bool, bool] | None, infile: JDFTXInfile):
    pbc_infile = get_pbc_from_infile(infile)
    if pbc is None:
        return get_pbc_from_infile(infile)
    elif pbc_infile is None:
        return pbc
    else:
        if not all([pbc[i] == pbc_infile[i] for i in range(3)]):
            raise ValueError(f"Given pbc {pbc} does not match pbc from infile {pbc_infile}")
        return pbc

from sys import exc_info

try:
    oid = read_opt_inputs()
    use_jdft = False
    work_dir = Path(oid["work_dir"])
    structure = oid["structure"]
    restart = oid["restart"]
    lat_iters = oid["lat_iters"]
    gpu = oid["gpu"]
    pbc = oid["pbc"]
    bias = oid["bias"]
    ortho = oid["ortho"]
    max_steps = oid["max_steps"]
    ddec6 = oid["ddec6"]
    pseudoSet = oid["pseudoSet"]
    freeze_base = oid["freeze_base"]
    freeze_tol = oid["freeze_tol"]
    freeze_count = oid["freeze_count"]
    exclude_freeze_count = oid["exclude_freeze_count"]
    direct_coords = oid["direct_coords"]
    freeze_map = oid["freeze_map"]
    freeze_all_but_map = oid["freeze_all_but_map"]
    if exclude_freeze_count > freeze_count:
        raise ValueError(f"freeze_count ({freeze_count}) must be greater than exclude_freeze_count ({exclude_freeze_count})")
    fmax = oid["fmax"]
    os.chdir(work_dir)
    opt_dir = work_dir / "ion_opt"
    lat_dir = work_dir / "lat_opt"
    structure = work_dir / structure
    opt_dir = str(opt_dir)
    structure = str(structure)
    lat_dir = str(lat_dir)
    opt_log = get_log_fn(str(work_dir), "opt", False, restart=restart, parallel=True)
    opt_log(f"Given opt_input: {oid}")
    apply_freeze_func = get_apply_freeze_func(freeze_base, freeze_tol, freeze_count, None, exclude_freeze_count, freeze_map=freeze_map, freeze_all_but_map=freeze_all_but_map, log_fn=opt_log)
    #opt_log(f"main: {freeze_idcs}")
    opt_log("Getting structure and restart status")
    structure = check_structure(structure, str(work_dir), log_fn=opt_log)
    opt_log(f"Structure: {structure}")
    atoms, restart = get_atoms(structure, restart, str(work_dir), opt_dir, lat_dir, lat_iters, use_jdft, log_fn=opt_log)
    # exe_cmd = get_exe_cmd(gpu, opt_log, use_srun=not debug)
    opt_log("getting cmds dict")
    cmds = get_cmds_dict(str(work_dir), ref_struct=structure, log_fn=opt_log, pbc=pbc, bias=bias)
    opt_log(f"cmds dict: {cmds}")
    cmds = cmds_dict_to_list(cmds)
    opt_log(f"Setting {structure} to atoms object")
    cmds = add_cohp_cmds(cmds, ortho=ortho)
    if ddec6:
        cmds = add_elec_density_dump(cmds)
    opt_log(f"Final cmds list: {cmds}")
    base_infile = cmds_list_to_infile(cmds)
    pbc = check_pbc(pbc, base_infile)
    atoms.pbc = pbc
    restarting_lat = False
    restarting_ion = (not restarting_lat) and (not ope(opj(opt_dir, "finished.txt")))
    restarting_ion = restarting_ion and restart
    opt_log(f"Running ion optimization with ASE optimizer")
    opt_log("Importing pyjdftx")
    import pyjdftx
    opt_log("Importing mpi4py")
    from mpi4py import MPI
    opt_log("Applying freeze constraints to atoms object")
    atoms_obj = apply_freeze_func(atoms)
    calc_dir = Path(opt_dir) / "jdftx_run"
    opt_log(f"Creating directory for pyjdftx calculation: {calc_dir}")
    calc_dir.mkdir(exist_ok=True, parents=True)
    outfile = calc_dir / "out"
    if not outfile.exists():
        opt_log(f"Writing initial out file for pyjdftx at {outfile}")
        with open(outfile, 'a') as f:
            f.write("creating file\n")
    opt_log("Initializing pyjdftx")
    # pyjdftx.initialize(MPI.COMM_WORLD, MPI.COMM_WORLD, "ion_opt/jdftx_run/out", False)
    pyjdftx.initialize(MPI.COMM_WORLD, MPI.COMM_WORLD, "ion_opt/jdftx_run/out", True)
    opt_log("Creating calculator object")
    kwargs = translate_infile_to_pydftx_kwargs(base_infile, {})
    kwargs["pseudopotentials"] = pseudoSet
    kwargs["commands"] = str(strip_infile_of_reserved_commands(base_infile))
    calculator_object = pyjdftx.ase.JDFTx(
        directory="ion_opt/jdftx_run",
        label="jdftx",
        **kwargs
    )
    opt_log(f"Setting calculator to atoms object")
    atoms_obj.set_calculator(calculator_object)
    opt_log("ASE ionic optimization starting")
    FIRE_kwargs = {
        "a": (150 / 70) * 0.1
    }
    kwargs.pop("opt_alpha", None)
    traj = opj(opt_dir, "opt.traj")
    log = opj(opt_dir, "opt.log")
    restart = opj(opt_dir, "hessian.pckl")
    FIRE_kwargs.update({"restart": restart})
    # if is_head():
    if True:
        opt_log("Running in head node - attaching trajectory and log file to optimizer kwargs")
        opt_log("trajectory path: " + traj)
        opt_log("log file path: " + log)
        FIRE_kwargs.update({"trajectory": traj, "logfile": log})
    
    dyn = FIRE(atoms, **FIRE_kwargs)
    ##
    opt_log("Optimization starting")
    opt_log(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    opt_log(f"Finished in {dyn.nsteps}/{max_steps}")
    finished(opt_dir)
    calculator_object.dump_end()
    pyjdftx.finalize(True)
    ###
    opt_log("Optimization finished.")
    if ddec6 and (is_head()):
        opt_log(f"Running DDEC6 analysis in {opt_dir}")
        run_ddec6(calc_dir, file_prefix="jdftx.")
except Exception as e:
    print(f"Error: {e}", file=stderr)
    print(exc_info())
    exit(1)

