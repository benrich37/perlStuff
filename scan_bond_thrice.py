import os
from os.path import join as opj
from os.path import exists as ope
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from JDFTx import JDFTx
import numpy as np
import shutil
from helpers.generic_helpers import optimizer, read_pbc_val, get_inputs_list, add_bond_constraints, get_log_fn, \
    get_cmds, write_contcar
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.generic_helpers import dump_template_input, get_nrg

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

bond_scan_template = ["scan: 3, 5, 10, 0.23",
                   "restart: 3",
                   "max_steps: 100",
                   "fmax: 0.05",
                   "follow: False",
                   "pbc: True, true, false"]


def read_scan_inputs(fname="scan_input"):
    """ Example:
    Scan: 1, 4, 10, -.2
    restart_at: 0
    work: /pscratch/sd/b/beri9208/1nPt1H_NEB/calcs/surfs/H2_H2O_start/No_bias/scan_bond_test/
    follow_momentum: True

    notes:
    Scan counts your atoms starting from 0, so the numbering in vesta will be 1 higher than what you need to put in here
    The first two numbers in scan are the two atom indices involved in the bond you want to scan
    The third number is the number of steps
    The fourth number is the step size (in angstroms) for each step
    """
    lookline = None
    restart_idx = 0
    work_dir = None
    follow = False
    pbc = [True, True, False]
    if not ope(fname):
        dump_template_input(fname, bond_scan_template, os.getcwd())
        raise ValueError(f"No bond scan input supplied: dumping template {fname}")
    inputs = get_inputs_list(fname)
    for input in inputs:
        key, val = input[0], input[1]
        if "scan" in key:
            lookline = val.split(",")
        if "restart" in key:
            restart_idx = int(val)
        if "work" in key:
            work_dir = val
        if "follow" in key:
            follow = "true" in val
        if "pbc" in key:
            pbc = read_pbc_val(val)
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    return atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, pbc


def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write("Done")


def prep_input(step_idx, atom_pair, step_length, step_type, start_length):
    target_length = start_length + (step_idx * step_length)
    atoms = read(str(step_idx) + "/CONTCAR", format="vasp")
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    dir_vec *= step_length/np.linalg.norm(dir_vec)
    if step_type == 0:
        atoms.positions[atom_pair[1]] += dir_vec
    elif step_type == 1:
        dir_vec *= 0.5
        atoms.positions[atom_pair[1]] += dir_vec
        atoms.positions[atom_pair[0]] += (-1) * dir_vec
    elif step_type == 2:
        atoms.positions[atom_pair[0]] += (-1)*dir_vec
    write(str(step_idx) + "/" + str(step_type) + "/POSCAR", atoms, format="vasp")


def copy_files(src_dir, tgt_dir):
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, tgt_dir)

def run_step(step_dir, fix_pair, pbc, log_fn, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(step_dir, "POSCAR"), format="vasp")
    atoms.pbc = pbc
    add_bond_constraints(atoms, fix_pair, log_fn=log_fn)
    calculator = get_calc(step_dir)
    print("setting calculator")
    atoms.set_calculator(calculator)
    print("printing atoms")
    print(atoms)
    print("setting optimizer")
    dyn = optimizer(atoms, step_dir, FIRE)
    traj = Trajectory(opj(step_dir,'opt.traj'), 'w', atoms, properties=['energy', 'forces'])
    print("attaching trajectory")
    dyn.attach(traj.write, interval=1)
    dyn.attach(lambda: write_contcar(atoms, step_dir), interval=1)
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        finished(step_dir)
    except Exception as e:
        print("couldnt run??")
        print(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)

def get_start_dist(work_dir, atom_pair):
    start_dir = os.path.join(work_dir, "0")
    atoms = read(os.path.join(start_dir, "CONTCAR"))
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    return np.linalg.norm(dir_vec)


if __name__ == '__main__':
    debug = False
    atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, pbc = read_scan_inputs()
    thrice_log = get_log_fn(work_dir, "scan_thrice", False)
    cmds = get_cmds(work_dir)
    exe_cmd = get_exe_cmd(True, thrice_log)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, log_fn=thrice_log)
    os.chdir(work_dir)
    if (not os.path.exists("./0")) or (not os.path.isdir("./0")):
        os.mkdir("./0")
    copy_files("./", "./0")
    start_length = get_start_dist(work_dir, atom_pair)
    for i in list(range(scan_steps))[restart_idx:]:
        if (not os.path.exists(f"./{str(i)}")) or (not os.path.isdir(f"./{str(i)}")):
            os.mkdir(f"./{str(i)}")
        if i > 0:
            copy_files(f"./{str(i-1)}", f"./{str(i)}")
        fs_cur = []
        for j in range(3):
            if not os.path.exists(f'{str(i)}/{str(j)}'):
                if not os.path.isdir(f'{str(i)}/{str(j)}'):
                    os.mkdir(f'{str(i)}/{str(j)}')
            copy_files(f"./{str(i)}", f"./{str(i)}/{str(j)}")
            prep_input(i, atom_pair, step_length, j, start_length)
            if not debug:
                run_step(f'{str(i)}/{str(j)}/', atom_pair, pbc, thrice_log, fmax=0.1, max_steps=50)
            fs_cur.append(get_nrg(f"./{str(i)}/{str(j)}/"))
        best_j = fs_cur.index(np.min(fs_cur))
        copy_files(f"./{str(i)}/{best_j}", f"./{str(i)}")
