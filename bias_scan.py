import os
from os.path import exists as ope, join as opj
from os import mkdir
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.units import Hartree
from datetime import datetime

from helpers.bias_scan_helpers import init_dir_name
from helpers.generic_helpers import get_inputs_list, fix_work_dir, optimizer, remove_dir_recursive, \
    get_atoms_list_from_out, get_do_cell, add_freeze_surf_base_constraint, get_cmds_dict, append_key_val_to_cmds_list
from helpers.generic_helpers import _write_contcar, get_log_fn, dump_template_input, read_pbc_val, SHE_work_val_Ha, get_mu
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.generic_helpers import check_submit, get_atoms_from_coords_out, add_cohp_cmds, get_atoms_from_out, add_elec_density_dump
from helpers.generic_helpers import copy_best_state_files, has_coords_out_files, get_lattice_cmds_list, get_ionic_opt_cmds_list
from helpers.generic_helpers import _write_opt_iolog, check_for_restart, log_def, check_structure, log_and_abort, cmds_dict_to_list
from helpers.logx_helpers import out_to_logx, _write_logx, finished_logx, sp_logx, opt_dot_log_faker
from scripts.run_ddec6 import main as run_ddec6
from sys import exit, stderr
from shutil import copy as cp
import numpy as np


opt_template = ["structure: POSCAR # Structure for optimization",
                "fmax: 0.04 # Max force for convergence criteria",
                "max steps: 100 # Max number of steps before exit",
                "gpu: True # Whether or not to use GPU (much faster)",
                "restart: True # Whether to get structure from lat/opt dirs or from input structure",
                "pbc: True True False # Periodic boundary conditions for unit cell",
                "lattice steps: 0 # Number of steps for lattice optimization (0 = no lattice optimization)",
                "freeze base: False # Whether to freeze lower atoms (don't use for bulk calcs)",
                "freeze tol: 3. # Distance from topmost atom to impose freeze cutoff for freeze base",
                "pseudoset: GBRV # directory name containing pseudopotentials you wish to use (top directory must be assigned to 'JDFTx_pseudo' environmental variable)",
                "ddec6: True # Dump Elec Density and perform DDEC6 analysis on it",
                "bmin: -1.0 # Min point of bias scan",
                "bmax: 1.0 # Max point of bias scan",
                "bsteps: 10 # Number of biases to evaluate",
                "bref: SHE # Ref for bias value, 'SHE' for wrt SHE, 'mu' for absolute bias, any number for custom ref (will be read in Hartree)",
                "scale: V # Scale for bais values, 'V' for eV, 'mu' for Hartrees",
                "init bias: PZC # Bias to initialize bias scan, 'PZC' for charge neutral, any number for a specific bias",
                "init ion opt: True # Whether to optimize geometry alongside initialization step",
                "init lat opt: False # Whether to optimize lattice alongside initialization step"]


def read_bias_scan_inputs(fname ="bias_scan_input"):
    work_dir = None
    structure = None
    if not ope(fname):
        dump_template_input(fname, opt_template, os.getcwd())
        raise ValueError(f"No bias scan input supplied: dumping template {fname}")
    inputs = get_inputs_list(fname, auto_lower=False)
    fmax = 0.01
    max_steps = 100
    gpu = True
    restart = False
    pbc = [True, True, False]
    lat_iters = 0
    freeze_base = False
    freeze_tol = 3.
    save_state = False
    ortho = True
    pseudoset = "GBRV"
    ddec6 = True
    bmin = -1
    bmax = 1
    bsteps = 10
    brefval = SHE_work_val_Ha
    bscale = "V"
    init_pzc = False
    init_bias = None
    init_ion_opt = True
    init_lat_opt = False
    for input in inputs:
        key, val = input[0], input[1]
        if "bmin" in key:
            bmin = float(val)
        elif "bmax" in key:
            bmax = float(val)
        elif "bstep" in key:
            bsteps = int(val)
        elif "bref" in key:
            brefval = read_bref_val(val)
        elif "init" in key:
            if "bias" in key:
                if "pzc" in val.lower():
                    init_pzc = True
                else:
                    init_bias = float(val)
            elif "opt" in key:
                if "ion" in key:
                    init_ion_opt = "true" in val.lower()
                elif "lat" in key:
                    init_lat_opt = "true" in val.lower()
                else:
                    print(f"Error reading {key}. Currently {init_ion_opt} for init ion opt and {init_lat_opt} for init lat opt")
        elif "pseudo" in key:
            pseudoset = val.strip()
        elif "structure" in key:
            structure = val.strip()
        elif "work" in key:
            work_dir = val
        elif "gpu" in key:
            gpu = "true" in val.lower()
        elif "restart" in key:
            restart = "true" in val.lower()
        elif "max" in key:
            if "fmax" in key:
                fmax = float(val)
            elif "step" in key:
                max_steps = int(val)
        elif "pbc" in key:
            pbc = read_pbc_val(val)
        elif "lat" in key:
            try:
                n_iters = int(val)
                lat_iters = n_iters
            except:
                pass
        elif ("freeze" in key):
            if ("base" in key):
                freeze_base = "true" in val.lower()
            elif ("tol" in key):
                freeze_tol = float(val)
        elif ("ortho" in key):
            ortho = "true" in val.lower()
        elif ("save" in key) and ("state" in key):
            save_state = "true" in val.lower()
        elif "ddec6" in key:
            ddec6 = "true" in val.lower()
    work_dir = fix_work_dir(work_dir)
    brange = get_mu_range(bmin, bmax, bsteps, brefval, bscale)
    return work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters, freeze_base, freeze_tol, ortho, save_state, pseudoset, ddec6, brange, init_pzc, init_bias, init_ion_opt, init_lat_opt


def read_bref_val(val):
    ref_strs = {"SHE": SHE_work_val_Ha,
                "mu": 0.0}
    ref_val = None
    read_float = False
    for k in ref_strs:
        if k.lower() in val.lower():
            ref_val = ref_strs[k]
        else:
            break
    if ref_val is None:
        ref_val = float(val)
    return ref_val

def define_dir(root, dirname):
    dirpath = opj(root, dirname)
    if not ope(dirpath):
        mkdir(dirpath)
    return dirpath

def get_mu_range(bmin, bmax, bsteps, brefval, bscale):
    ev_to_mu = (-1/Hartree)
    if bscale.lower() in ["v", "ev"]:
        bmin *= ev_to_mu
        bmax *= ev_to_mu
    bmin -= brefval
    bmax -= brefval
    brange = np.linspace(bmin, bmax, bsteps)
    return brange




def finished(dirname):
    with open(opj(dirname, "finished.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")

def is_finished(dirname):
    return ope(opj(dirname, "finished.txt"))


def get_atoms_from_lat_dir(dir):
    ionpos = opj(dir, "ionpos")
    lattice = opj(dir, "lattice")
    return get_atoms_from_coords_out(ionpos, lattice)

def make_dir(dirname):
    if not ope(dirname):
        os.mkdir(dirname)



def get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
    no_opt_struc = False
    no_lat_struc = False
    if ope(opt_dir):
        log_fn(f"{opt_dir} exists - checking this directory first for restart structure")
        if not use_jdft:
            if ope(opj(opt_dir, "CONTCAR")):
                structure = opj(opt_dir, "CONTCAR")
                log_fn(f"Found {structure} for restart structure")
        else:
            outfile = opj(opt_dir, "out")
            if ope(outfile):
                try:
                    atoms_obj = get_atoms_from_out(outfile)
                    structure = opj(opt_dir, "POSCAR")
                    write(structure, atoms_obj, format="vasp")
                except:
                    no_opt_struc = True
    else:
        no_opt_struc = True
    if ope(lat_dir) and no_opt_struc:
        make_dir(opt_dir)
        if not has_coords_out_files(lat_dir):
            log_fn(f"No ionpos and/or lattice found in {lat_dir}")
            lat_out = opj(lat_dir, "out")
            if ope(lat_out):
                log_fn(f"Reading recent structure from out file in {lat_out}")
                atoms = get_atoms_list_from_out(lat_out)[-1]
                structure = opj(lat_dir, "POSCAR_lat_out")
                log_fn(f"Saving read structure to {structure}")
                write(structure, atoms, format="vasp")
        else:
            log_fn(f"Reading structure from {lat_dir}")
            atoms = get_atoms_from_lat_dir(lat_dir)
            structure = opj(lat_dir, "POSCAR_coords_out")
            log_fn(f"Saving read structure to {structure}")
            write(structure, atoms, format="vasp")
    else:
        no_lat_struc = True
    if no_lat_struc:
        make_dir(lat_dir)
        make_dir(opt_dir)
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    return structure, restart


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





def run_lat_opt_runner(atoms, structure, lat_dir, root, calc_fn, log_fn=log_def):
    atoms.set_calculator(calc_fn(lat_dir))
    log_fn("lattice optimization starting")
    atoms.get_forces()
    ionpos = opj(lat_dir, "ionpos")
    lattice = opj(lat_dir, "lattice")
    pbc = atoms.pbc
    atoms = get_atoms_from_coords_out(ionpos, lattice)
    atoms.pbc = pbc
    structure = opj(lat_dir, "CONTCAR")
    write(structure, atoms, format="vasp")
    log_fn(f"Finished lattice optimization")
    finished(lat_dir)
    return atoms, structure


def run_lat_opt(atoms, structure, lat_dir, root, calc_fn, log_fn=log_def, _failed_before=False):
    run_again = False
    try:
        atoms, structure = run_lat_opt_runner(atoms, structure, lat_dir, root, calc_fn, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, lat_dir, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms, structure = run_lat_opt(atoms, structure, lat_dir, root, calc_fn, log_fn=log_fn, _failed_before=True)
    return atoms, structure


def run_ion_opt_runner(atoms_obj, ion_dir_path, calc_fn, freeze_base = False, freeze_tol = 0., log_fn=log_def):
    add_freeze_surf_base_constraint(atoms_obj, ztol=freeze_tol, freeze_base=freeze_base)
    atoms_obj.set_calculator(calc_fn(ion_dir_path))
    log_fn("ionic optimization starting")
    pbc = atoms_obj.pbc
    atoms_obj.get_forces()
    outfile = opj(ion_dir_path, "out")
    if ope(outfile):
        atoms_obj_list = get_atoms_list_from_out(outfile)
        atoms_obj = atoms_obj_list[-1]
    else:
        log_and_abort(f"No output data given - check error file", log_fn=log_fn)
    atoms_obj.pbc = pbc
    structure_path = opj(ion_dir_path, "CONTCAR")
    write(structure_path, atoms_obj, format="vasp")
    structure_path = opj(ion_dir_path, "CONTCAR.gjf")
    write(structure_path, atoms_obj, format="gaussian-in")
    finished(ion_dir_path)
    return atoms_obj


def run_ion_opt(atoms_obj, ion_dir_path, calc_fn, freeze_base = False, freeze_tol = 0., _failed_before=False, log_fn=log_def):
    run_again = False
    try:
        atoms_obj = run_ion_opt_runner(atoms_obj, ion_dir_path, calc_fn, freeze_base = freeze_base, freeze_tol = freeze_tol, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, ion_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms_obj = run_ion_opt(atoms_obj, ion_dir_path, calc_fn, _failed_before=True, log_fn=log_fn)
    return atoms_obj

def copy_result_files(opt_dir, work_dir):
    result_files = ["CONTCAR", "CONTCAR.gjf", "Ecomponents", "out"]
    for file in result_files:
        full = opj(opt_dir, file)
        if ope(full):
            cp(full, work_dir)


def append_out_to_logx(outfile, logx, log_fn=log_def):
    log_fn(f"Gathering minimization structures from existing out file")
    atoms_list = get_atoms_list_from_out(outfile)
    log_fn(f"{len(atoms_list)} found from out file")
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

def make_pzc_cmds(cmds):
    cmds_new = []
    for cmd in cmds:
        if not "target-mu" in cmd[0]:
            cmds_new.append(cmd)
    return cmds_new


def run_init(init_dir, atoms, cmds, init_pzc, init_bias, init_ion_opt, init_lat_opt, pbc, exe_cmd, pseudoSet, freeze_base=False, freeze_tol=0.0, log_fn=log_def):
    log_fn("Setting up initialization calc")
    if not init_pzc:
        cmds = append_key_val_to_cmds_list(cmds, "target-mu", str(init_bias), allow_duplicates = False)
    if init_lat_opt:
        log_fn("Setting up lattice optimization of initialization calc")
        lat_cmds = get_lattice_cmds_list(cmds, 100, pbc)
        lat_dir = define_dir(init_dir, "lat_opt")
        get_lat_calc = lambda root: _get_calc(exe_cmd, lat_cmds, root, pseudoSet=pseudoSet, log_fn=log_fn)
        log_fn("Running lattice optimization of initialization calc")
        run_lat_opt(atoms, None, lat_dir, None, get_lat_calc, log_fn=log_fn)
    if init_ion_opt:
        log_fn("Setting up ionic optimization of initialization calc")
        ion_cmds = get_ionic_opt_cmds_list(cmds, 100)
    else:
        log_fn("Setting up single point calculation for initialization")
        ion_cmds = get_ionic_opt_cmds_list(cmds, 0)
    ion_dir = define_dir(init_dir, "ion_opt")
    get_ion_calc = lambda root: _get_calc(exe_cmd, ion_cmds, root, pseudoSet=pseudoSet, log_fn=log_fn)
    log_fn("Running initialization calc")
    run_ion_opt(atoms, ion_dir, get_ion_calc, freeze_base = freeze_base, freeze_tol = freeze_tol, log_fn=log_fn)
    log_fn("Initialization calc finished")


def get_init_mu(scan_dir):
    init_dir = opj(scan_dir, init_dir_name)
    outfname = opj(init_dir, opj("ion_opt", "out"))
    mu = get_mu(outfname)
    return mu

def make_scan_dirs(scan_dir, brange):
    nsteps = len(brange)
    stepdirs = []
    completed = []
    for i in range(nsteps):
        stepdir = define_dir(scan_dir, str(i))
        stepdirs.append(stepdir)
        completed.append(False)
    return stepdirs, completed

def run_scan_step(idx, step_dirs, ref_dir, completed, scan_step_runner, log_fn=log_def):
    try:
        scan_step_runner(step_dirs[idx], ref_dir)
        completed[idx] = True
        log_fn(f"Scan step at {step_dirs[idx]} completed.")
    except Exception as e:
        log_fn(e)
        pass
    return completed

def get_ref_dir(mu_eval, murange, stepdirs, completed_bools, init_dir, log_fn=log_def):
    idcs = np.argsort([abs(mu_eval - mu) for mu in murange])
    if not True in completed_bools:
        log_fn(f"No completed scan steps found. Will use init dir {init_dir} for reference")
        return init_dir
    else:
        for idx in idcs:
            if completed_bools[idx]:
                log_fn(f"Closest mu to {mu_eval} of finished calcs found to be {[abs(mu_eval - mu) for mu in murange][idx]} in {stepdirs[idx]}")
                return stepdirs[idx]


def _scan_step_runner(step_dir, ref_dir, fmax, max_steps, pbc, lat_iters, pseudoset, cmds, exe_cmd, ddec6, freeze_base=False, freeze_tol=0.0, log_fn=log_def):
    ref_dir_calc_dir = opj(ref_dir, "ion_opt")
    atoms = get_atoms_from_out(opj(ref_dir_calc_dir, "out"))
    ion_dir = define_dir(step_dir, "ion_opt")
    if not is_finished(ion_dir):
        if lat_iters > 0:
            lat_dir = define_dir(step_dir, "lat_opt")
            if not is_finished(lat_dir):
                copy_best_state_files([ref_dir_calc_dir], lat_dir, log_fn)
                lat_cmds = get_lattice_cmds_list(cmds, 100, pbc)
                get_lat_calc = lambda root: _get_calc(exe_cmd, lat_cmds, root, pseudoSet=pseudoset, log_fn=log_fn)
                run_lat_opt(atoms, None, lat_dir, None, get_lat_calc, log_fn=log_fn)
                copy_best_state_files([lat_dir], ion_dir, log_fn)
        else:
            copy_best_state_files([ref_dir_calc_dir], ion_dir, log_fn)
        ion_cmds = get_ionic_opt_cmds_list(cmds, max_steps)
        get_ion_calc = lambda root: _get_calc(exe_cmd, ion_cmds, root, pseudoSet=pseudoset, log_fn=log_fn)
        run_ion_opt(atoms, ion_dir, get_ion_calc, freeze_base=freeze_base, freeze_tol=freeze_tol, log_fn=log_fn)
        if ddec6:
            log_fn(f"Running DDEC6 analysis in {ion_dir}")
            run_ddec6(ion_dir)



def run_scan(scan_dir, brange, cmds, fmax, max_steps, pbc, lat_iters, pseudoset, exe_cmd, ddec6, freeze_base=False, freeze_tol=0.0, log_fn=log_def):
    log_fn("Setting up bias scan")
    log_fn(f"Bias range is {brange} in abs Hartree")
    init_dir = opj(scan_dir, init_dir_name)
    init_mu = get_init_mu(scan_dir)
    step_dirs, completed = make_scan_dirs(scan_dir, brange)
    log_fn(f"Sorting scan steps w.r.t. initialization mu (= {init_mu:.4f} in abs Ha)")
    bdifs = [abs(b - init_mu) for b in brange]
    idcs = np.argsort(bdifs)
    run_order = [list(range(len(brange)))[idx] for idx in idcs]
    log_fn(f"Bias scan steps will be run in order {run_order}")
    for i, idx in enumerate(run_order):
        log_fn(f"Setting up scan step {idx}")
        cmds = append_key_val_to_cmds_list(cmds, "target-mu", str(brange[idx]), allow_duplicates=False, log_fn=log_fn)
        scan_step_runner = lambda calc_dir, ref_dir: _scan_step_runner(step_dir, ref_dir, fmax, max_steps, pbc, lat_iters, pseudoset, cmds, exe_cmd, ddec6, freeze_base=freeze_base, freeze_tol=freeze_tol, log_fn=log_fn)
        step_dir = step_dirs[idx]
        ref_dir = get_ref_dir(brange[idx], brange, step_dirs, completed, init_dir, log_fn=log_fn)
        log_fn(f"Running step {idx} ({step_dir})")
        completed = run_scan_step(idx, step_dirs, ref_dir, completed, scan_step_runner, log_fn=log_fn)
        if completed[idx]:
            log_fn(f"Scan step {idx} successfully completed")
        else:
            log_fn(f"Scan step {idx} did not complete successfully. Ignoring and moving on.")
    log_fn("Bias scan completed.")





def main():
    work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters, freeze_base, freeze_tol, ortho, \
        save_state, pseudoset, ddec6, brange, init_pzc, init_bias, init_ion_opt, init_lat_opt = read_bias_scan_inputs()
    os.chdir(work_dir)
    scan_dir = define_dir(work_dir, "bias_scan")
    init_dir = define_dir(scan_dir, init_dir_name)
    structure = opj(work_dir, structure)
    scan_log = get_log_fn(work_dir, "bias_scan", False, restart=restart)
    structure = check_structure(structure, work_dir, log_fn=scan_log)
    exe_cmd = get_exe_cmd(gpu, scan_log)
    atoms = read(structure, format="vasp")
    cmds = get_cmds_dict(work_dir, ref_struct=structure, pbc=pbc, log_fn=scan_log)
    cmds = cmds_dict_to_list(cmds)
    cmds = make_pzc_cmds(cmds)
    cmds = add_cohp_cmds(cmds, ortho=ortho)
    if ddec6:
        cmds = add_elec_density_dump(cmds)
    check_submit(gpu, os.getcwd(), "bias_scan", log_fn=scan_log)
    run_init(init_dir, atoms, cmds, init_pzc, init_bias, init_ion_opt, init_lat_opt, pbc, exe_cmd, pseudoset, freeze_base=freeze_base, freeze_tol=freeze_tol, log_fn=scan_log)
    run_scan(scan_dir, brange, cmds, fmax, max_steps, pbc, lat_iters, pseudoset, exe_cmd, ddec6, freeze_base=freeze_base, freeze_tol=freeze_tol, log_fn=scan_log)

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

