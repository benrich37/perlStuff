import os
from os.path import exists as ope, join as opj
from ase.io import read, write as _write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase import Atoms, Atom
from ase.constraints import FixAtoms
from datetime import datetime
from helpers.generic_helpers import get_cmds_list, get_inputs_list, fix_work_dir, optimizer, remove_dir_recursive, \
    get_atoms_list_from_out, get_do_cell, add_freeze_surf_base_constraint, get_cmds_dict
from helpers.generic_helpers import _write_contcar, get_log_fn, dump_template_input, read_pbc_val
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.generic_helpers import check_submit, get_atoms_from_coords_out, add_cohp_cmds, get_atoms_from_out, add_elec_density_dump
from helpers.generic_helpers import copy_best_state_files, has_coords_out_files, get_lattice_cmds_list, get_ionic_opt_cmds_list
from helpers.generic_helpers import _write_opt_iolog, check_for_restart, log_def, check_structure, log_and_abort, cmds_dict_to_list
from helpers.logx_helpers import out_to_logx, _write_logx, finished_logx, sp_logx, opt_dot_log_faker
from scripts.run_ddec6 import main as run_ddec6
from sys import exit, stderr
from shutil import copy as cp
from os import getcwd
import numpy as np
import subprocess

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
        "pbc": [True, True, False],
        "lat_iters": 0,
        "use_jdft": True,
        "freeze_base": False,
        "freeze_tol": 3.,
        "ortho": True,
        "save_state": False,
        "pseudoSet": "GBRV",
        "bias": 0.0,
        "ddec6": True,
        "freeze_count": 0,
        "exclude_freeze_count": 0,
        "direct_coords": False
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


def finished(dirname):
    with open(opj(dirname, "finished.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")


def get_atoms_from_lat_dir(dir):
    outfile = opj(dir, "out")
    return get_atoms_from_out(outfile)

def make_dir(dirname):
    if not ope(dirname):
        os.mkdir(dirname)



def get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_def):
    atoms_obj = None
    outfile_path = opj(opt_dir, "out")
    if ope(outfile_path):
        try:
            atoms_obj = get_atoms_from_out(outfile_path)
            log_fn(f"Atoms object set from {outfile_path}")
        except Exception as e:
            log_fn(f"Error reading atoms object from {outfile_path}")
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
    

        


# def get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
#     no_opt_struc = False
#     no_lat_struc = False
#     if ope(opt_dir):
#         log_fn(f"{opt_dir} exists - checking this directory first for restart structure")
#         if not use_jdft:
#             if ope(opj(opt_dir, "CONTCAR")):
#                 structure = opj(opt_dir, "CONTCAR")
#                 log_fn(f"Found {structure} for restart structure")
#         else:
#             outfile = opj(opt_dir, "out")
#             if ope(outfile):
#                 try:
#                     atoms_obj = get_atoms_from_out(outfile)
#                     structure = opj(opt_dir, "POSCAR")
#                     write(structure, atoms_obj, format="vasp")
#                 except:
#                     no_opt_struc = True
#     else:
#         no_opt_struc = True
#     if ope(lat_dir) and no_opt_struc:
#         make_dir(opt_dir)
#         if not has_coords_out_files(lat_dir):
#             log_fn(f"No ionpos and/or lattice found in {lat_dir}")
#             lat_out = opj(lat_dir, "out")
#             if ope(lat_out):
#                 log_fn(f"Reading recent structure from out file in {lat_out}")
#                 atoms = get_atoms_list_from_out(lat_out)[-1]
#                 structure = opj(lat_dir, "POSCAR_lat_out")
#                 log_fn(f"Saving read structure to {structure}")
#                 write(structure, atoms, format="vasp")
#         else:
#             log_fn(f"Reading structure from {lat_dir}")
#             atoms = get_atoms_from_lat_dir(lat_dir)
#             structure = opj(lat_dir, "POSCAR_coords_out")
#             log_fn(f"Saving read structure to {structure}")
#             write(structure, atoms, format="vasp")
#     else:
#         no_lat_struc = True
#     if no_lat_struc and no_opt_struc:
#         make_dir(lat_dir)
#         make_dir(opt_dir)
#         log_fn(f"Could not gather restart structure from {work_dir}")
#         if ope(structure):
#             log_fn(f"Using {structure} for structure")
#             log_fn(f"Changing restart to False")
#             restart = False
#             log_fn("setting up lattice and opt dir")
#         else:
#             log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
#     return structure, restart


def get_structure(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=log_def):
    dirs_list = [opt_dir]
    if lat_iters > 0:
        log_fn(f"Lattice opt requested ({lat_iters} iterations) - adding lat dir to setup list")
        dirs_list.append(lat_dir)
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





def run_lat_opt_runner(
        atoms: Atoms, structure: str, lat_dir: str, root: str, calc_fn,
        freeze_base = False, freeze_tol = 0., freeze_count = 0, log_fn=log_def
        ):
    add_freeze_surf_base_constraint(atoms, ztol=freeze_tol, freeze_base=freeze_base, freeze_count = freeze_count)
    atoms.set_calculator(calc_fn(lat_dir))
    log_fn("lattice optimization starting")
    atoms.get_forces()
    log_fn("lattice optimization finished - organizing output data")
    outfile = opj(lat_dir, "out")
    pbc = atoms.pbc
    if ope(outfile):
        atoms_obj: Atoms = get_atoms_from_out(outfile)
    else:
        log_and_abort(f"No output data given - check error file", log_fn=log_fn)
    atoms_obj.pbc = pbc
    structure_path = opj(lat_dir, "CONTCAR")
    write(structure_path, atoms_obj, format="vasp")
    structure_path = opj(lat_dir, "CONTCAR.gjf")
    write(structure_path, atoms_obj, format="gaussian-in")
    finished(lat_dir)
    return atoms, structure


def run_lat_opt(atoms, structure, lat_dir, root, calc_fn, freeze_base = False, freeze_tol = 0., freeze_count = 0, log_fn=log_def, _failed_before=False):
    run_again = False
    try:
        atoms, structure = run_lat_opt_runner(atoms, structure, lat_dir, root, calc_fn, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, lat_dir, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms, structure = run_lat_opt(atoms, structure, lat_dir, root, calc_fn, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=log_fn, _failed_before=True)
    return atoms, structure


def run_ion_opt_runner(atoms_obj: Atoms, ion_dir_path, calc_fn, freeze_base = False, freeze_tol = 0., freeze_count = 0, exclude_freeze_count=0,  log_fn=log_def):
    add_freeze_surf_base_constraint(atoms_obj, ztol=freeze_tol, freeze_base=freeze_base,  freeze_count = freeze_count, exclude_freeze_count=exclude_freeze_count)
    calculator_object = calc_fn(ion_dir_path)
    atoms_obj.set_calculator(calculator_object)
    log_fn("ionic optimization starting")
    pbc = atoms_obj.pbc
    atoms_obj.get_forces()
    log_fn("ionic optimization finished - organizing output data")
    outfile = opj(ion_dir_path, "out")
    if ope(outfile):
        atoms_obj = get_atoms_from_out(outfile)
    else:
        log_and_abort(f"No output data given - check error file", log_fn=log_fn)
    atoms_obj.pbc = pbc
    structure_path = opj(ion_dir_path, "CONTCAR")
    write(structure_path, atoms_obj, format="vasp")
    structure_path = opj(ion_dir_path, "CONTCAR.gjf")
    write(structure_path, atoms_obj, format="gaussian-in")
    finished(ion_dir_path)
    return atoms_obj


def run_ion_opt(atoms_obj, ion_dir_path, root_path, calc_fn, freeze_base = False, freeze_tol = 0., freeze_count = 0, exclude_freeze_count=0, _failed_before=False, log_fn=log_def):
    run_again = False
    try:
        atoms_obj = run_ion_opt_runner(atoms_obj, ion_dir_path, calc_fn, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, exclude_freeze_count=exclude_freeze_count, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, ion_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms_obj = run_ion_opt(atoms_obj, ion_dir_path, root_path, calc_fn,
                                freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count,exclude_freeze_count=exclude_freeze_count, _failed_before=True, log_fn=log_fn)
    return atoms_obj


def run_ase_opt_runner(atoms, root, opter, fmax, max_steps, freeze_base = False, freeze_tol = 0., freeze_count = 0,log_fn=log_def):
    add_freeze_surf_base_constraint(atoms, ztol=freeze_tol, freeze_base=freeze_base, freeze_count = freeze_count)
    do_cell = get_do_cell(atoms.pbc)
    dyn = optimizer(atoms, root, opter)
    traj = Trajectory(opj(root, "opt.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
    logx = opj(root, "opt.logx")
    write_logx = lambda: _write_logx(atoms, logx, do_cell=do_cell)
    write_contcar = lambda: _write_contcar(atoms, root)
    write_opt_log = lambda: _write_opt_iolog(atoms, dyn, max_steps, log_fn)
    dyn.attach(traj.write, interval=1)
    dyn.attach(write_contcar, interval=1)
    dyn.attach(write_logx, interval=1)
    dyn.attach(write_opt_log, interval=1)
    log_fn("Optimization starting")
    log_fn(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
    finished_logx(atoms, logx, dyn.nsteps, max_steps)
    sp_logx(atoms, "sp.logx", do_cell=do_cell)
    finished(root)


def run_ase_opt(atoms, opt_dir, opter, calc_fn, fmax, max_steps, freeze_base = False, freeze_tol = 0., freeze_count = 0,log_fn=log_def, _failed_before=False):
    atoms.set_calculator(calc_fn(opt_dir))
    run_again = False
    try:
        run_ase_opt_runner(atoms, opt_dir, opter, fmax, max_steps, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, opt_dir, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        run_ase_opt(atoms, opt_dir, opter, calc_fn, fmax, max_steps, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=log_fn, _failed_before=True)

def copy_result_files(opt_dir, work_dir):
    result_files = ["CONTCAR", "CONTCAR.gjf", "Ecomponents", "out"]
    for file in result_files:
        full = opj(opt_dir, file)
        if ope(full):
            cp(full, work_dir)


def append_out_to_logx(outfile, logx, log_fn=log_def):
    log_fn(f"Gathering minimization structures from existing out file")
    _atoms_list = get_atoms_list_from_out(outfile)
    atoms_list = []
    for atoms in _atoms_list:
        if not atoms is None:
            atoms_list.append(atoms)
    log_fn(f"{len(atoms_list)} found from out file")
    if len(atoms_list):
        do_cell = get_do_cell(atoms_list[-1].pbc)
        for atoms in atoms_list:
            _write_logx(atoms, logx, do_cell=do_cell)

def make_jdft_logx(opt_dir, log_fn=log_def):
    outfile = opj(opt_dir, "out")
    logx = opj(opt_dir, "out.logx")
    if ope(logx):
        log_fn("Appending existing out file data to existing logx file")
        append_out_to_logx(outfile, logx, log_fn=log_fn)
    elif ope(outfile):
        out_to_logx(opt_dir, outfile, log_fn=log_fn)
        log_fn("Writing existing out file to logx file")



def outfile_protect(work_dir):
    calc_dirs = [opj(work_dir, t) for t in ["ion_opt", "lat_opt"]]
    outfiles = [opj(c, "out") for c in calc_dirs]
    for o in outfiles:
        if ope(o):
            if not debug:
                print("DELETING FLUID-EX-CORR LINE FROM FUNCTIONAL - DELETE ME ONCE THIS BUG IS FIXED")
                subprocess.run(f"sed -i '/fluid-ex-corr/d' {o}", shell=True, check=True)

def fix_restart_bug(work_dir, restart):
    outfile = opj(opj(work_dir, "ion_opt"), "out")
    if restart:
        if ope(outfile):
            if not debug:
                subprocess.run(f"sed -i '/fluid-ex-corr/d' {outfile}", shell=True, check=True)
                subprocess.run(f"sed -i '/lda-PZ/d' {outfile}", shell=True, check=True)

def main():
    # work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters, use_jdft, freeze_base, freeze_tol, ortho, save_state, pseudoSet, bias, ddec6 = read_opt_inputs()
    oid = read_opt_inputs()
    work_dir = oid["work_dir"]
    outfile_protect(work_dir)
    structure = oid["structure"]
    restart = oid["restart"]
    fix_restart_bug(work_dir, restart)
    lat_iters = oid["lat_iters"]
    use_jdft = oid["use_jdft"]
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
    if exclude_freeze_count > freeze_count:
        raise ValueError(f"freeze_count ({freeze_count}) must be greater than exclude_freeze_count ({exclude_freeze_count})")
    fmax = oid["fmax"]
    os.chdir(work_dir)
    opt_dir = opj(work_dir, "ion_opt")
    lat_dir = opj(work_dir, "lat_opt")
    structure = opj(work_dir, structure)
    opt_log = get_log_fn(work_dir, "opt", False, restart=restart)
    structure = check_structure(structure, work_dir, log_fn=opt_log)
    structure, restart = get_structure(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=opt_log)
    exe_cmd = get_exe_cmd(gpu, opt_log)
    cmds = get_cmds_dict(work_dir, ref_struct=structure, bias=bias, pbc=pbc, log_fn=opt_log)
    # cmds = get_cmds_list(work_dir, ref_struct=structure)
    opt_log(f"Setting {structure} to atoms object")
    atoms = read(structure, format="vasp")
    atoms.pbc = pbc
    # cmds = add_dos_cmds(cmds, atoms, save_dos, save_pdos)
    cmds = cmds_dict_to_list(cmds)
    cmds = add_cohp_cmds(cmds, ortho=ortho)
    lat_cmds = get_lattice_cmds_list(cmds, lat_iters, pbc)
    ion_cmds = get_ionic_opt_cmds_list(cmds, max_steps)
    if ddec6:
        ion_cmds = add_elec_density_dump(ion_cmds)
    
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, pseudoSet=pseudoSet, log_fn=opt_log)
    get_lat_calc = lambda root: _get_calc(exe_cmd, lat_cmds, root, pseudoSet=pseudoSet, log_fn=opt_log, direct_coords=direct_coords)
    get_ion_calc = lambda root: _get_calc(exe_cmd, ion_cmds, root, pseudoSet=pseudoSet, log_fn=opt_log)
    check_submit(gpu, os.getcwd(), "opt", log_fn=opt_log)
    lat_finished = ope(opj(lat_dir, "finished.txt"))
    do_lat = (lat_iters > 0) and (not lat_finished)
    restarting_lat = do_lat and restart
    if do_lat:
        if restarting_lat:
            make_jdft_logx(lat_dir, log_fn=opt_log)
        atoms, structure = run_lat_opt(atoms, structure, lat_dir, work_dir, get_lat_calc, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=opt_log)
        make_jdft_logx(lat_dir, log_fn=opt_log)
        opt_dot_log_faker(opj(lat_dir, "out"), lat_dir)
        cp(opj(lat_dir, "opt.log"), work_dir)
    restarting_ion = (not restarting_lat) and (not ope(opj(opt_dir, "finished.txt")))
    restarting_ion = restarting_ion and restart
    opt_log(f"Finding/copying any state files to {opt_dir}")
    copy_best_state_files([work_dir, lat_dir, opt_dir], opt_dir, log_fn=opt_log)
    if use_jdft:
        if restarting_ion:
            make_jdft_logx(opt_dir, log_fn=opt_log)
        opt_log(f"Running ion optimization with JDFTx optimizer")
        run_ion_opt(atoms, opt_dir, work_dir, get_ion_calc, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=opt_log, exclude_freeze_count=exclude_freeze_count)
        make_jdft_logx(opt_dir, log_fn=opt_log)
        opt_dot_log_faker(opj(opt_dir, "out"), opt_dir)
        if not (lat_iters > 0):
            cp(opj(opt_dir, "opt.log"), work_dir)
    else:
        opt_log(f"Running ion optimization with ASE optimizer")
        run_ase_opt(atoms, opt_dir, FIRE, get_calc, fmax, max_steps, freeze_base = freeze_base, freeze_tol = freeze_tol, freeze_count = freeze_count, log_fn=opt_log)
    opt_log("Optimization finished.")
    if ddec6:
        opt_log("Running DDEC6 analysis")
        run_ddec6(opt_dir, pbc=pbc)
    # copy_result_files(opt_dir, work_dir)

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

