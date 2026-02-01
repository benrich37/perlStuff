# For using pyjdftx with ASE optimizers

import os
from os.path import exists as ope, join as opj
from ase.io import read, write as _write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase import Atoms, Atom
from ase.constraints import FixAtoms
from datetime import datetime
# from DihedralScan import DihedralScan
from RotationScan import RotationScan
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
        "init_step": 0.,
        "gpu": True,
        "dangle": 10.0,
        "nsteps": 18,
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
        if "pbc" in key:
            opt_inputs_dict["pbc"] = read_pbc_val(val)
        if ("direct" in key) and ("coord" in key):
            opt_inputs_dict["direct_coords"] = "true" in val.lower()
        if ("mol" in key) and ("idcs" in key):
            opt_inputs_dict["mol_idcs"] = parse_dihedral_idcs(val)[0]
        if "center" in key:
            opt_inputs_dict["center"] = parse_dihedral_idcs(val)[0]
        if "axis" in key:
            opt_inputs_dict["axis"] = parse_dihedral_idcs(val)[0]
        if ("mask" in key):
            opt_inputs_dict["mask_list"] = parse_dihedral_idcs(val)
        if ("init" in key):
            opt_inputs_dict["init_step"] = float(val)
        if ("nstep" in key):
            opt_inputs_dict["nsteps"] = int(val)
        if ("dangle" in key):
            opt_inputs_dict["dangle"] = float(val)
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

def parse_dihedral_idcs_entry(entry: str):
    if "none" in entry.lower():
        return None
    return [int(v.strip()) for v in entry.split(',')]

def parse_dihedral_idcs(line: str):
    # Expected format: (i1, i2, i3, i4), (i1, i2, i3, i4), ...
    pieces = []
    start_idcs = [i for i, v in enumerate(line) if v == "("]
    end_idcs = [i for i, v in enumerate(line) if v == ")"]
    if len(start_idcs):
        for start, end in zip(start_idcs, end_idcs):
            pieces.append(line[start + 1:end].strip())
    else:
        pieces = [line.strip()]
    dihedral_list = []
    for piece in pieces:
        dihedral_list.append(parse_dihedral_idcs_entry(piece))
    if len(dihedral_list) == 1:
        return dihedral_list[0]
    return dihedral_list


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
    if not atoms is None:
        structure = opj(opt_dir, "POSCAR")
        write(structure, atoms, format="vasp")
        return structure, restart
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(lat_dir, log_fn=log_fn)
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
    for path in [opt_dir, lat_dir]:
        if not ope(path):
            make_dir(path)
    # If an atoms is found in the ion_opt dir, then an ionic minimization was run
    # (So even if the lattice opt was run, the ion opt will be the most recent)
    atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn)
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




# def run_ase_opt_runner(atoms, root, opter, fmax, max_steps, freeze_base = False, freeze_tol = 0., freeze_count = 0,log_fn=log_def):
#     add_freeze_surf_base_constraint(atoms, ztol=freeze_tol, freeze_base=freeze_base, freeze_count = freeze_count)
#     do_cell = get_do_cell(atoms.pbc)
#     dyn = optimizer(atoms, root, opter)
#     traj = Trajectory(opj(root, "opt.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
#     logx = opj(root, "opt.logx")
#     write_logx = lambda: _write_logx(atoms, logx, do_cell=do_cell)
#     write_contcar = lambda: _write_contcar(atoms, root)
#     write_opt_log = lambda: _write_opt_iolog(atoms, dyn, max_steps, log_fn)
#     dyn.attach(traj.write, interval=1)
#     dyn.attach(write_contcar, interval=1)
#     dyn.attach(write_logx, interval=1)
#     dyn.attach(write_opt_log, interval=1)
#     log_fn("Optimization starting")
#     log_fn(f"Fmax: {fmax}, max_steps: {max_steps}")
#     dyn.run(fmax=fmax, steps=max_steps)
#     log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
#     finished_logx(atoms, logx, dyn.nsteps, max_steps)
#     sp_logx(atoms, "sp.logx", do_cell=do_cell)
#     finished(root)


def run_rotation_scan(atoms_obj, ion_dir_path, calc_fn, 
                      nsteps: int, dangle: float, 
                      mol_idcs: list[int],
                      axis: list[int],
                      center: list[int] | int,
                      init_step: float = 0.,
                      log_fn=log_def
                      ):
    import pyjdftx
    calculator_object = calc_fn(ion_dir_path)
    atoms_obj.set_calculator(calculator_object)
    log_fn("ASE ionic optimization starting")
    rot_scan_kwargs = {
        "dangle": dangle,
        "total_steps": nsteps,
        "mol_idcs": mol_idcs,
        "axis": axis,
        "center": center,
        "init_step": init_step,
    }
    dyn = optimizer(atoms_obj, ion_dir_path, RotationScan, 
                    opt_alpha=None,
                    **rot_scan_kwargs,
                    )
    log_fn("Scan starting")
    log_fn(", ".join([f"{k}: {v}" for k, v in rot_scan_kwargs.items()]))
    # log_fn(f"Dangle: {dangle}, nsteps: {nsteps}, mol_idcs: {dihedral_list}, mask_list: {mask_list}")
    dyn.run()
    log_fn(f"Finished in {dyn.nsteps}/{nsteps}")
    finished(ion_dir_path)
    # calculator_object.dump_end()
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
    
def get_proper_mask_list_entry(mask_list_input, n_atoms, log_fn=log_def):
    if mask_list_input is None:
        return None
    elif isinstance(mask_list_input[0], int): # Only one mask list
        proper_mask = np.zeros(n_atoms)
        for idx in mask_list_input:
            proper_mask[idx] = 1
        return proper_mask.tolist()
    
def get_proper_mask_list(mask_list_input, n_atoms, log_fn=log_def):
    if isinstance(mask_list_input, list) and (not isinstance(mask_list_input[0], int)):
        proper_mask_list = []
        for mask_entry in mask_list_input:
            proper_mask = get_proper_mask_list_entry(mask_entry, n_atoms, log_fn=log_fn)
            proper_mask_list.append(proper_mask)
        return proper_mask_list
    else:
        return get_proper_mask_list_entry(mask_list_input, n_atoms, log_fn=log_fn)

def main(debug=False):
    oid = read_opt_inputs()
    use_jdft = False
    work_dir = Path(oid["work_dir"])
    structure = oid["structure"]
    restart = oid["restart"]
    gpu = oid["gpu"]
    pbc = oid["pbc"]
    bias = oid["bias"]
    ddec6 = oid["ddec6"]
    pseudoSet = oid["pseudoSet"]
    direct_coords = oid["direct_coords"]
    nsteps = oid["nsteps"]
    dangle = oid["dangle"]
    init_step = oid["init_step"]
    

    mol_idcs = oid["mol_idcs"]

    os.chdir(work_dir)
    opt_dir = work_dir / "scan_dihedral"
    structure = work_dir / structure
    opt_dir = str(opt_dir)
    structure = str(structure)
    lat_dir = str(opt_dir)
    opt_log = get_log_fn(str(work_dir), "rot_scan", False, restart=restart)
    opt_log(f"Given opt_input: {oid}")
    structure = check_structure(structure, str(work_dir), log_fn=opt_log)
    atoms, restart = get_atoms(structure, restart, str(work_dir), opt_dir, lat_dir, 0, use_jdft, log_fn=opt_log)
    cmds = get_cmds_dict(str(work_dir), ref_struct=structure, log_fn=opt_log, pbc=pbc, bias=bias)
    cmds = cmds_dict_to_list(cmds)
    opt_log(f"Setting {structure} to atoms object")
    if ddec6:
        cmds = add_elec_density_dump(cmds)
    base_infile = cmds_list_to_infile(cmds)
    pbc = check_pbc(pbc, base_infile)
    atoms.pbc = pbc
    get_arb_calc = lambda root, cmds: get_calc_pyjdftx(cmds, root, pseudoSet=pseudoSet, debug=debug, log_fn=opt_log, direct_coords=direct_coords)
    get_calc = lambda root: get_arb_calc(root, base_infile)
    check_submit(gpu, os.getcwd(), "opt", log_fn=opt_log)
    restarting_ion = (not ope(opj(opt_dir, "finished.txt")))
    restarting_ion = restarting_ion and restart
    opt_log(f"Running ion optimization with ASE optimizer")
    run_rotation_scan(atoms, opt_dir, get_calc, nsteps, dangle, mol_idcs, oid.get("axis", None), oid.get("center", None), init_step=init_step, log_fn=opt_log)
    # run_dihedral_scan(atoms, opt_dir, get_calc, nsteps, dangle, dihedral_list, mask_list=mask_list, init_step=init_step, log_fn=opt_log)
    opt_log("Optimization finished.")

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

