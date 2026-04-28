# For using pyjdftx with ASE optimizers

import os
from os.path import exists as ope, join as opj
from ase.io import read, write as _write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase import Atoms, Atom
from ase.constraints import FixAtoms
from datetime import datetime
from helpers.generic_helpers import get_cmds_list, get_inputs_list, fix_work_dir, optimizer, remove_dir_recursive, \
    get_atoms_list_from_out, get_do_cell, add_freeze_surf_base_constraint, get_cmds_dict, get_apply_freeze_func
from helpers.generic_helpers import _write_contcar, get_log_fn, dump_template_input, read_pbc_val
from helpers.calc_helpers import _get_calc, get_exe_cmd, _get_calc_new, get_calc_pyjdftx
from helpers.generic_helpers import check_submit, get_atoms_from_coords_out, add_cohp_cmds, get_atoms_from_out, add_elec_density_dump
from helpers.generic_helpers import copy_best_state_files, has_coords_out_files, get_lattice_cmds_list, get_ionic_opt_cmds_list
from helpers.generic_helpers import _write_opt_iolog, check_for_restart, log_def, check_structure, log_and_abort, cmds_dict_to_list, cmds_list_to_infile
from helpers.logx_helpers import out_to_logx, _write_logx, finished_logx, sp_logx, opt_dot_log_faker
from scripts.run_ddec6_v3 import main as run_ddec6
from sys import exit, stderr
from shutil import copy as cp
from os import getcwd
import numpy as np
import subprocess
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path

cwd = getcwd()
debug = "perlStuff" in cwd

def write(fname, _atoms, format="vasp"):
    atoms = _atoms.copy()
    atoms.pbc = [True,True,True]
    _write(fname, atoms, format=format)



opt_template = ["structure: POSCAR # Structure for optimization",
                "fmax: 0.04 # Max force for convergence criteria",
                "max_steps: 100 # Max number of steps before exit",
                "gpu: True # Whether or not to use GPU (much faster)",
                "restart: True # Whether to get structure from lat/opt dirs or from input structure",
                "pbc: True True False # Periodic boundary conditions for unit cell",
                "lattice steps: 0 # Number of steps for lattice optimization (0 = no lattice optimization)",
                "opt program: jdft # Which program to use for ionic optimization, options are 'jdft' and 'ase'",
                "freeze base: False # Whether to freeze lower atoms (don't use for bulk calcs)",
                "freeze tol: 3. # Distance from topmost atom to impose freeze cutoff for freeze base",
                "freeze count: 0 # freeze the lowest n atoms - overrides freeze tol if greater than 0"
                "pseudoset: GBRV # directory name containing pseudopotentials you wish to use (top directory must be assigned to 'JDFTx_pseudo' environmental variable)",
                "bias: 0.00V # Bias relative to SHE (is only used if 'target-mu *' in inputs file",
                "ddec6: True # Dump Elec Density and perform DDEC6 analysis on it"]



def read_opt_inputs(fname = "opt_input"):
    if not ope(fname):
        dump_template_input(fname, opt_template, os.getcwd())
        raise ValueError(f"No opt input supplied: dumping template {fname}")
    inputs = get_inputs_list(fname, auto_lower=False)
    opt_inputs_dict = {
        "work_dir": None,
        "structure": None,
        "fmax": 0.01,
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
    opt_inputs_dict["work_dir"] = fix_work_dir(opt_inputs_dict["work_dir"])
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

def finished(dirname, suffix=""):
    with open(opj(dirname, f"finished{suffix}.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")

def finished_label(dirname, label):
    finished(dirname, suffix=f"_{label}")

def is_finished(dirname, suffix=""):
    return ope(opj(dirname, f"finished{suffix}.txt"))

def label_is_finished(dirname, label):
    return is_finished(dirname, suffix=f"_{label}")




def run_ase_opt(atoms_obj, ion_dir_path, opter, calc_fn, fmax, max_steps, apply_freeze_func, log_fn=log_def, _failed_before=False):
    log_fn("Import pyjdftx")
    import pyjdftx
    log_fn("Creating calculator object")
    calculator_object = calc_fn(ion_dir_path)
    log_fn(f"Setting calculator to atoms object")
    atoms_obj.set_calculator(calculator_object)
    atoms_obj = apply_freeze_func(atoms_obj)
    log_fn("ASE ionic optimization starting")
    dyn = optimizer(atoms_obj, ion_dir_path, opter)
    log_fn("Optimization starting")
    log_fn(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
    finished(ion_dir_path)
    calculator_object.dump_end()
    pyjdftx.finalize(True)



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
    
def get_atomss(calc_root, prefix="POSCAR_", suffix=".gjf", read_format="gaussian-in", log_fn=log_def) -> tuple[list[Atoms], list[str]]:
    files = [f for f in os.listdir(calc_root) if f.startswith(prefix) and f.endswith(suffix)]
    try:
        files = sorted(files, key=lambda x: int(x[len(prefix):-len(suffix)]))
    except:
        log_fn(f"Warning: non-integer labels found in {calc_root} with prefix {prefix} and suffix {suffix}, using unsorted file order: {files}")
    atomss = []
    labels = []
    for f in files:
        atoms = read(opj(calc_root, f), format=format)
        atomss.append(atoms)
        labels.append(f[len(prefix):-len(suffix)])
    log_fn(f"Found {len(atomss)} files with prefix {prefix} and suffix {suffix} in {calc_root} with labels {labels}")
    atomss, labels = update_atomss(calc_root, atomss, labels, prefix=prefix, suffix=suffix, read_format=read_format, log_fn=log_fn)
    return atomss, labels

def trim_finished_atoms(calc_root, atomss, labels, prefix="CONTCAR_", suffix=".gjf", read_format="gaussian-in", log_fn=log_def) -> tuple[list[Atoms], list[str]]:
    files = [f for f in os.listdir(calc_root) if f.startswith(prefix) and f.endswith(suffix)]
    finished_labels = [f[len(prefix):-len(suffix)] for f in files]
    atomss_trimmed = []
    labels_trimmed = []
    for atoms, label in zip(atomss, labels):
        if label in finished_labels:
            log_fn(f"Excluding {label} from optimization list (is finished)")
        else:
            atomss_trimmed.append(atoms)
            labels_trimmed.append(label)
    return atomss_trimmed, labels_trimmed

def update_atomss(calc_root, atomss, labels, prefix="CONTCAR_", suffix=".gjf", read_format="gaussian-in", log_fn=log_def) -> tuple[list[Atoms], list[str]]:
    atomss, labels = trim_finished_atoms(calc_root, atomss, labels, prefix=prefix, suffix=suffix, read_format=read_format, log_fn=log_fn)
    working_label_idcs = [i for i, label in enumerate(labels) if label_is_working(calc_root / "serial_opt", label)]
    if len(working_label_idcs) > 1:
        log_fn(f"Warning: multiple working labels found in {calc_root}/serial_opt: {[labels[i] for i in working_label_idcs]}. Ignoring working structures and using original POSCAR_*.gjf files for all optimizations.")
    elif len(working_label_idcs) == 1:
        log_fn(f"Updating atoms for working label {labels[working_label_idcs[0]]} from serial_opt directory")
        atomss[working_label_idcs[0]] = get_working_atoms(calc_root / "serial_opt")
    else:
        log_fn(f"No working labels found in {calc_root}/serial_opt, using original POSCAR_*.gjf files for all optimizations")
    return atomss, labels

def log_results(ion_dir_path: Path, atoms_obj, label: str, log_fn=log_def):
    log_fn(f"Logging results for {label}")
    label_results_path = ion_dir_path / label
    label_results_path.mkdir(exist_ok=True)
    cur_result_files = [f for f in os.listdir(ion_dir_path / "jdftx_run")]
    for f in cur_result_files:
        src = ion_dir_path / "jdftx_run" / f
        dst = label_results_path / f
        log_fn(f"Copying {src} to {dst}")
        cp(src, dst)
    write(label_results_path / f"CONTCAR_{label}.gjf", atoms_obj, format="gaussian-in")
    write(ion_dir_path.parent / f"CONTCAR_{label}.gjf", atoms_obj, format="gaussian-in")
    finished(label_results_path)

def working_label(ion_dir_path: Path, label: str, log_fn=log_def):
    with open(ion_dir_path / f"working_{label}.txt", 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Working")

def label_is_working(ion_dir_path: Path, label: str):
    return ope(ion_dir_path / f"working_{label}.txt")

def get_working_atoms(ion_dir_path, log_fn=log_def) -> Atoms:
    outfile_path = ion_dir_path / "jdftx_run" / "jdftx.out"
    outfile = JDFTXOutfile.from_file(outfile_path)
    working_structure = outfile.structure
    return AseAtomsAdaptor.get_atoms(working_structure)

def update_atoms(atoms_obj, atoms_new, log_fn=log_def):
    atoms_obj.set_positions(atoms_new.get_positions())
    return atoms_obj

def opt_serial(calc_root: Path, atoms_obj: Atoms, atomss: list[Atoms], labelss: list[str], opter, calc_fn, fmax, max_steps, apply_freeze_func, log_fn=log_def):
    log_fn("Import pyjdftx")
    import pyjdftx
    ion_dir_path = calc_root / "serial_opt"
    ion_dir_path.mkdir(exist_ok=True)
    log_fn("Creating calculator object")
    calculator_object = calc_fn(ion_dir_path)
    log_fn(f"Setting calculator to atoms object")
    atoms_obj.set_calculator(calculator_object)
    atoms_obj = apply_freeze_func(atoms_obj)
    for i, label in enumerate(labelss):
        atoms_obj = update_atoms(atoms_obj, atomss[i], log_fn=log_fn)
        dyn = optimizer(atoms_obj, ion_dir_path, opter)
        working_label(ion_dir_path, label, log_fn=log_fn)
        dyn.run(fmax=fmax, steps=max_steps)
        log_results(ion_dir_path, atoms_obj, label, log_fn=log_fn)
        finished_label(ion_dir_path, label)
        log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
        calculator_object.dump_end() # Unsure if this can be inside the loop to allow dumping arrays and whatnot
    pyjdftx.finalize(True)

def write_dummy_structure(work_dir, atoms, sample_name="POSCAR_sample"):
    dummy_structure_path = Path(work_dir) / f"{sample_name}"
    write(dummy_structure_path, atoms, format="vasp")
    return dummy_structure_path


def main(debug=False):
    # work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters, use_jdft, freeze_base, freeze_tol, ortho, save_state, pseudoSet, bias, ddec6 = read_opt_inputs()
    oid = read_opt_inputs()
    use_jdft = False
    work_dir = Path(oid["work_dir"])
    restart = oid["restart"]
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
    opt_dir = work_dir / "serial_opt"
    opt_dir = str(opt_dir)
    opt_log = get_log_fn(str(work_dir), "serial_opt", False, restart=restart)
    opt_log(f"Given opt_input: {oid}")
    apply_freeze_func = get_apply_freeze_func(freeze_base, freeze_tol, freeze_count, None, exclude_freeze_count, freeze_map=freeze_map, freeze_all_but_map=freeze_all_but_map, log_fn=opt_log)
    # finished labels should already be excluded, and working labels should be updated to current structure
    atomss, labels = get_atomss(work_dir, prefix="POSCAR_", suffix=".gjf", read_format="gaussian-in")
    if not len(atomss):
        opt_log(f"No POSCAR_*.gjf files found in {work_dir} for trajectory recording - looking for POSCAR_* (vasp) files instead")
        atomss, labels = get_atomss(work_dir, prefix="POSCAR_", suffix="", format="vasp")
    atoms_obj = atomss[0].copy()
    structure = write_dummy_structure(work_dir, atoms_obj)
    cmds = get_cmds_dict(str(work_dir), ref_struct=structure, log_fn=opt_log, pbc=pbc, bias=bias)
    cmds = cmds_dict_to_list(cmds)
    cmds = add_cohp_cmds(cmds, ortho=ortho)
    if ddec6:
        cmds = add_elec_density_dump(cmds)
    base_infile = cmds_list_to_infile(cmds)
    atoms_obj.pbc = check_pbc(pbc, base_infile)
    get_arb_calc = lambda root, cmds: get_calc_pyjdftx(cmds, root, pseudoSet=pseudoSet, debug=debug, log_fn=opt_log, direct_coords=direct_coords)
    get_calc = lambda root: get_arb_calc(root, base_infile)
    # check_submit(gpu, os.getcwd(), "serial_opt", log_fn=opt_log)
    opt_serial(Path(work_dir), atoms_obj, atomss, labels, FIRE, get_calc, fmax, max_steps, apply_freeze_func, log_fn=opt_log)

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

