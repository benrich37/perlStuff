import os
from os.path import join as opj
from os.path import exists as ope
from ase.io import read, write
import subprocess
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import FIRE
from JDFTx import JDFTx
import numpy as np
import shutil
from generic_helpers import copy_rel_files, get_cmds, get_inputs_list, fix_work_dir, optimizer, remove_dir_recursive
from generic_helpers import _write_contcar, get_log_fn, dump_template_input, read_pbc_val, get_exe_cmd, _get_calc
from generic_helpers import _write_logx, finished_logx, check_submit, sp_logx
from scan_bond_helpers import _scan_log, _prep_input


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
                "pbc: False False False"]

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
    work_dir = fix_work_dir(work_dir)
    return work_dir, structure, fmax, max_steps, gpu, restart, pbc

def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write("Done")


if __name__ == '__main__':
    work_dir, structure, fmax, max_steps, gpu, restart, pbc = read_opt_inputs()
    check_submit(gpu, os.getcwd())
    opt_log = get_log_fn(work_dir, "opt_io", False)
    if restart:
        structure = "CONTCAR"
        opt_log("Requested restart: reading from CONTCAR in existing opt directory")
    exe_cmd = get_exe_cmd(gpu, opt_log)
    cmds = get_cmds(work_dir, ref_struct=structure)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, JDFTx, log_fn=opt_log)
    os.chdir(work_dir)
    opt_log(f"structure: {structure}")
    opt_log(f"work_dir: {work_dir}")
    opt_dir = opj(work_dir, "opt")
    if not restart:
        opt_log("setting up opt dir")
        if (not ope(opt_dir)) or (not os.path.isdir(opt_dir)):
            os.mkdir(opt_dir)
        else:
            remove_dir_recursive(opt_dir)
            os.mkdir(opt_dir)
        copy_rel_files("./", opt_dir)
        shutil.copy(opj(work_dir, structure), opt_dir)
    start_path = opj(opt_dir, structure)
    opt_log(f"Reading {start_path} for structure")
    atoms = read(opj(opt_dir, structure))
    atoms.pbc = pbc
    atoms.set_calculator(get_calc(opt_dir))
    dyn = optimizer(atoms, opt_dir, FIRE)
    traj = Trajectory(opj(opt_dir, "opt.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
    dyn.attach(traj.write, interval=1)
    write_contcar = lambda: _write_contcar(atoms, opt_dir)
    dyn.attach(write_contcar, interval=1)
    do_cell = True in pbc
    logx = "opt/opt.logx"
    write_logx = lambda: _write_logx(atoms, logx, dyn, max_steps, do_cell=do_cell)
    dyn.attach(write_logx, interval=1)
    opt_log("optimization starting")
    opt_log(f"Fmax: {fmax} \nmax_steps: {max_steps}")
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        opt_log(f"Finished in {dyn.nsteps}/{max_steps}\n")
        finished_logx(atoms, logx, dyn.nsteps, max_steps)
        sp_logx(atoms, "sp.logx", do_cell=do_cell)
        finished(opt_dir)
    except Exception as e:
        opt_log("couldnt run??")
        opt_log(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)
