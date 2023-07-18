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
from generic_helpers import insert_el, copy_rel_files, get_cmds, get_inputs_list, fix_work_dir, optimizer, add_bond_constraints, write_contcar, log_generic
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


"""
#!/bin/bash
#SBATCH -J scanny
#SBATCH --time=12:00:00
#SBATCH -o scanny.out
#SBATCH -e scanny.err
#SBATCH -q regular_ss11
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -A m4025_g

module use --append /global/cfs/cdirs/m4025/Software/Perlmutter/modules
module load jdftx/gpu

export JDFTx_NUM_PROCS=1

export SLURM_CPU_BIND="cores"
export JDFTX_MEMPOOL_SIZE=36000
export MPICH_GPU_SUPPORT_ENABLED=1

python /global/homes/b/beri9208/BEAST_DB_Manager/manager/scan_bond.py > scan.out
exit 0
"""

def read_opt_inputs(fname = "opt_input"):
    """ Example:
    structure: POSCAR
    """
    work_dir = None
    structure = None
    inputs = get_inputs_list(fname)
    fmax = 0.01
    max_steps = 100
    gpu = True
    restart = False
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
    work_dir = fix_work_dir(work_dir)
    return work_dir, structure, fmax, max_steps, gpu, restart

def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write("Done")

def get_calc(exe_cmd, cmds, calc_dir):
    return JDFTx(
        executable=exe_cmd,
        pseudoSet="GBRV_v1.5",
        commands=cmds,
        outfile=calc_dir,
        ionic_steps=False
    )


if __name__ == '__main__':
    work_dir, structure, fmax, max_steps, gpu, restart = read_opt_inputs()
    opt_log = lambda s: log_generic(s, work_dir, "opt_io", False)
    if ope("opt_io.log"):
        os.remove("opt_io.log")
    if restart:
        structure = "CONTCAR"
        opt_log("Requested restart: reading from CONTCAR in existing opt directory")
    if gpu:
        _get = 'JDFTx_GPU'
    else:
        _get = 'JDFTx'
    opt_log(f"Using {_get} for JDFTx exe")
    exe_cmd = 'srun ' + os.environ[_get]
    opt_log(f"exe cmd: {exe_cmd}")
    os.chdir(work_dir)
    opt_log(f"structure: {structure}")
    opt_log(f"work_dir: {work_dir}")
    cmds = get_cmds(work_dir, ref_struct = structure)
    opt_dir = opj(work_dir, "opt")
    if not restart:
        opt_log("setting up opt dir")
        if (not ope(opt_dir)) or (not os.path.isdir(opt_dir)):
            os.mkdir(opt_dir)
        copy_rel_files("./", opt_dir)
        shutil.copy(opj(work_dir, structure), opt_dir)
    atoms = read(opj(opt_dir, structure))
    atoms.set_calculator(get_calc(exe_cmd, opt_dir, cmds))
    dyn = optimizer(atoms, opt_dir, FIRE)
    traj = Trajectory(opj(opt_dir, "opt.traj"), 'w', atoms, properties=['energy', 'forces'])
    dyn.attach(traj.write, interval=1)
    write_contcar = lambda a: write_contcar(a, opt_dir)
    dyn.attach(write_contcar, interval=1)
    opt_log("optimization starting")
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        finished(opt_dir)
    except Exception as e:
        opt_log("couldnt run??")
        opt_log(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)
