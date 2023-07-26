import os
from os.path import join as opj
from os.path import exists as ope
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from JDFTx import JDFTx
import datetime
from generic_helpers import get_cmds, get_inputs_list, fix_work_dir, optimizer, remove_dir_recursive
from generic_helpers import _write_contcar, get_log_fn, dump_template_input, read_pbc_val, get_exe_cmd, _get_calc
from generic_helpers import _write_logx, finished_logx, check_submit, sp_logx, get_atoms_from_coords_out
from generic_helpers import copy_best_state_f, has_coords_out_files, get_lattice_cmds
from generic_helpers import out_to_logx, _write_opt_log, check_for_restart
from scripts.out_to_logx import get_do_cell, get_atoms_list_from_out


""" HOW TO USE ME:
- Be on perlmutter
- Go to the directory of the optimized geometry you want to perform a bond scan on
- Create a file called "scan_input" in that directory (see the read_scan_inputs function below for how to make that)
- Copied below is an example submit.sh to run it
- Make sure JDFTx.py's "constructInput" function is edited so that the "if" statement (around line 234) is given an
else statement that sets "vc = v + "\n""
- This script will read "CONTCAR" in the directory, and SAVES ANY ATOM FREEZING INFO (if you don't want this, make sure
either there is no "F F F" after each atom position in your CONTCAR or make sure there is only "T T T" after each atom
position (pro tip: sed -i 's/F F F/T T T/g' CONTCAR)
- This script checks first for your "inputs" file for info on how to run this calculation - if you want to change this
to a lower level (ie change kpoint-folding to 1 1 1), make sure you delete all the State output files from the directory
(wfns, fillings, etc)
"""


opt_template = ["structure: POSCAR_new #using this one bc idk",
                "fmax: 0.05",
                "max_steps: 30",
                "gpu: False",
                "restart: False",
                "pbc: False False False",
                "lattice steps: 0"]

def read_opt_inputs(fname = "opt_input"):
    """ Example:
    structure: POSCAR
    """
    work_dir = None
    structure = None
    if not ope(fname):
        dump_template_input(fname, opt_template, os.getcwd())
        raise ValueError(f"No opt input supplied: dumping template {fname}")
    inputs = get_inputs_list(fname, auto_lower=False)
    fmax = 0.01
    max_steps = 100
    gpu = True
    restart = False
    pbc = [True, True, False]
    lat_iters = 0
    for input in inputs:
        key, val = input[0], input[1]
        if "structure" in key:
            structure = val.strip()
        if "work" in key:
            work_dir = val
        if "gpu" in key:
            gpu = "true" in val.lower()
        if "restart" in key:
            restart = "true" in val.lower()
        if "max" in key:
            if "fmax" in key:
                fmax = float(val)
            elif "step" in key:
                max_steps = int(val)
        if "pbc" in key:
            pbc = read_pbc_val(val)
        if "lat" in key:
            try:
                n_iters = int(val)
                lat_iters = n_iters
            except:
                pass
    work_dir = fix_work_dir(work_dir)
    return work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters

def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")

def get_atoms_from_lat_dir(dir):
    ionpos = opj(dir, "ionpos")
    lattice = opj(dir, "lattice")
    return get_atoms_from_coords_out(ionpos, lattice)

def get_restart_structure(structure, restart, opt_dir, lat_dir, log_fn):
    if ope(opt_dir):
        if ope(opj(opt_dir, "CONTCAR")):
            structure = opj(opt_dir, "CONTCAR")
            log_fn(f"Found {structure} for restart structure")
    elif ope(lat_dir):
        os.mkdir(opt_dir)
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
        os.mkdir(lat_dir)
        os.mkdir(opt_dir)
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
        else:
            err = f"Requested structure {structure} not found"
            log_fn(err)
            raise ValueError(err)
    return structure, restart

def get_structure(structure, restart, opt_dir, lat_dir, opt_log):
    if not restart:
        for d in [opt_dir, lat_dir]:
            if ope(d):
                opt_log(f"Resetting {d}")
                remove_dir_recursive(d)
            os.mkdir(d)
        if ope(structure):
            opt_log(f"Found {structure} for structure")
        else:
            opt_log(f"Requested structure {structure} not found")
            raise ValueError("Missing input structure")
    else:
        structure, restart = get_restart_structure(structure, restart, opt_dir, lat_dir, opt_log)
    return structure, restart


def run_lat_opt_runner(atoms, structure, lat_iters, lat_dir, root, log_fn, cmds):
    lat_cmds = get_lattice_cmds(cmds, lat_iters, atoms.pbc)
    get_lat_calc = lambda root: _get_calc(exe_cmd, lat_cmds, root, JDFTx, log_fn=log_fn)
    atoms.set_calculator(get_lat_calc(lat_dir))
    log_fn("lattice optimization starting")
    log_fn(f"Fmax: n/a, max_steps: {lat_iters}")
    atoms.get_forces()
    ionpos = opj(lat_dir, "ionpos")
    lattice = opj(lat_dir, "lattice")
    pbc = atoms.pbc
    atoms = get_atoms_from_coords_out(ionpos, lattice)
    atoms.pbc = pbc
    structure = opj(root, structure + "_lat_opted")
    write(structure, atoms, format="vasp")
    opt_log(f"Finished lattice optimization")
    finished(lat_dir)
    out_to_logx(lat_dir, opj(lat_dir, 'out'), log_fn=log_fn)
    return atoms, structure

def run_lat_opt(atoms, structure, lat_iters, lat_dir, root, log_fn, cmds, _failed_before=False):
    run_again = False
    try:
        atoms, structure = run_lat_opt_runner(atoms, structure, lat_iters, lat_dir, root, log_fn, cmds)
    except Exception as e:
        assert check_for_restart(e, _failed_before, lat_dir, log_fn)
        run_again = True
        pass
    if run_again:
        atoms, structure = run_lat_opt(atoms, structure, lat_iters, lat_dir, root, log_fn, cmds, _failed_before=True)
    return atoms, structure


def run_ase_opt_runner(atoms, root, opter, do_cell, log_fn):
    dyn = optimizer(atoms, root, opter)
    traj = Trajectory(opj(root, "opt.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
    logx = opj(root, "opt.logx")
    write_logx = lambda: _write_logx(atoms, logx, dyn, max_steps, do_cell=do_cell)
    write_contcar = lambda: _write_contcar(atoms, root)
    write_opt_log = lambda: _write_opt_log(atoms, dyn, max_steps, log_fn)
    dyn.attach(traj.write, interval=1)
    dyn.attach(write_contcar, interval=1)
    dyn.attach(write_logx, interval=1)
    dyn.attach(write_opt_log, interval=1)
    opt_log("Optimization starting")
    opt_log(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    opt_log(f"Finished in {dyn.nsteps}/{max_steps}")
    finished_logx(atoms, logx, dyn.nsteps, max_steps)
    sp_logx(atoms, "sp.logx", do_cell=do_cell)
    finished(root)

def run_ase_opt(atoms, opt_dir, opter, do_cell, log_fn, exe_cmd, cmds, _failed_before = False):
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, JDFTx, log_fn=log_fn)
    atoms.set_calculator(get_calc(opt_dir))
    run_again = False
    try:
        run_ase_opt_runner(atoms, opt_dir, opter, do_cell, log_fn)
    except Exception as e:
        assert check_for_restart(e, _failed_before, opt_dir, log_fn)
        run_again = True
        pass
    if run_again:
        run_ase_opt(atoms, opt_dir, opter, do_cell, log_fn, exe_cmd, cmds, _failed_before=True)



if __name__ == '__main__':
    work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters = read_opt_inputs()
    check_submit(gpu, os.getcwd())
    os.chdir(work_dir)
    opt_dir = opj(work_dir, "opt")
    lat_dir = opj(work_dir, "lat")
    structure = opj(work_dir, structure)
    opt_log = get_log_fn(work_dir, "opt_io", False, restart=restart)
    structure, restart = get_structure(structure, restart, opt_dir, lat_dir, opt_log)
    exe_cmd = get_exe_cmd(gpu, opt_log)
    cmds = get_cmds(work_dir, ref_struct=structure)
    opt_log(f"Setting {structure} to atoms object")
    atoms = read(structure, format="vasp")
    do_cell = get_do_cell(pbc)
    atoms.pbc = pbc
    if (lat_iters > 0) and (not ope(opj(lat_dir,"finished.txt"))):
        atoms, structure = run_lat_opt(atoms, structure, lat_iters, lat_dir, work_dir, opt_log, cmds)
    copy_best_state_f([work_dir, lat_dir], opt_dir, log_fn=opt_log)
    run_ase_opt(atoms, opt_dir, FIRE, do_cell, opt_log, exe_cmd, cmds)
