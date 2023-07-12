import os
from ase.io import read, write
import subprocess
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE
from JDFTx import JDFTx
import numpy as np
import shutil
from generic_helpers import insert_el, copy_rel_files, get_cmds
from scan_bond_helpers import read_scan_inputs, get_start_dist, _scan_log, _prep_input


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



def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write("Done")


def optimizer(atoms, opt="FIRE", opt_alpha=150, logfile='opt.log'):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    opt_dict = {'BFGS': BFGS, 'BFGSLineSearch': BFGSLineSearch,
                'LBFGS': LBFGS, 'LBFGSLineSearch': LBFGSLineSearch,
                'GPMin': GPMin, 'MDMin': MDMin, 'FIRE': FIRE}
    if opt in ['BFGS', 'LBFGS']:
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='hessian.pckl', alpha=opt_alpha)
    elif opt == 'FIRE':
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='hessian.pckl', a=(opt_alpha / 70) * 0.1)
    else:
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='hessian.pckl')
    return dyn

def bond_constraint(atoms, indices):
    atoms.set_constraint(FixBondLength(indices[0], indices[1]))
    return atoms


def set_calc(exe_cmd, cmds, calc_dir):
    return JDFTx(
        executable=exe_cmd,
        pseudoSet="GBRV_v1.5",
        commands=cmds,
        outfile=calc_dir,
        ionic_steps=False
    )

def prep_input(step_idx, atom_pair, step_length):
    atoms = read(str(step_idx) + "/CONTCAR", format="vasp")
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    dir_vec *= step_length/np.linalg.norm(dir_vec)
    atoms.positions[atom_pair[1]] += dir_vec
    write(str(step_idx) + "/POSCAR", atoms, format="vasp")

def prep_input_alt(step_idx, atom_pair, step_length, start_length):
    target_length = start_length + (step_idx*step_length)
    atoms_recent = read(str(step_idx - 1) + "/CONTCAR", format="vasp")
    atoms_prev = read(str(step_idx - 2) + "/CONTCAR", format="vasp")
    dir_vecs = []
    for i in range(len(atoms_recent.positions)):
        dir_vecs.append(atoms_recent[i] - atoms_prev[i])
    for i in range(len(dir_vecs)):
        atoms_recent.positions[i] += dir_vecs[i]
    dir_vec = atoms_recent.positions[atom_pair[1]] - atoms_recent.positions[atom_pair[0]]
    cur_length = np.linalg.norm(dir_vec)
    should_be_0 = target_length - cur_length
    if not np.isclose(should_be_0, 0.0):
        atoms_recent.positions[atom_pair[1]] += dir_vec*(should_be_0)
    write(str(step_idx) + "/POSCAR", atoms_recent, format="vasp")


def run_step(step_dir, fix_pair, exe_cmd, inputs_cmds, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(step_dir, "POSCAR"), format="vasp")
    atoms.pbc = [True, True, False]
    bond_constraint(atoms, fix_pair)
    scan_log("creating calculator")
    calculator = set_calc(exe_cmd, step_dir, inputs_cmds)
    scan_log("setting calculator")
    atoms.set_calculator(calculator)
    scan_log("printing atoms")
    scan_log(atoms)
    scan_log("setting optimizer")
    dyn = optimizer(atoms, logfile=os.path.join(step_dir, "opt.log"))
    traj = Trajectory(step_dir +'opt.traj', 'w', atoms, properties=['energy', 'forces'])
    scan_log("attaching trajectory")
    dyn.attach(traj.write, interval=1)
    def write_contcar(a=atoms):
        a.write(step_dir +'CONTCAR', format="vasp", direct=True)
        insert_el(step_dir +'CONTCAR')
    dyn.attach(write_contcar, interval=1)
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        finished(step_dir)
    except Exception as e:
        scan_log("couldnt run??")
        scan_log(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)


if __name__ == '__main__':
    jdftx_exe = os.environ['JDFTx_GPU']
    exe_cmd = 'srun ' + jdftx_exe
    atom_pair, scan_steps, step_length, restart_idx, work_dir, follow = read_scan_inputs()
    scan_log = lambda s: _scan_log(s, work_dir)
    start_length = get_start_dist(work_dir, atom_pair)
    prep_input = lambda s, a, l: _prep_input(s, a, l, start_length, follow, scan_log, work_dir, 1)
    os.chdir(work_dir)
    cmds = get_cmds(work_dir)
    if (not os.path.exists("./0")) or (not os.path.isdir("./0")):
        os.mkdir("./0")
    copy_rel_files("./", "./0")
    for i in list(range(scan_steps))[restart_idx:]:
        if (not os.path.exists(f"./{str(i)}")) or (not os.path.isdir(f"./{str(i)}")):
            os.mkdir(f"./{str(i)}")
        if i > 0:
            copy_rel_files(f"./{str(i-1)}", f"./{str(i)}")
        if (i > 1):
            prep_input(i, atom_pair, step_length)
        run_step(work_dir + str(i) +"/", atom_pair, exe_cmd, cmds, fmax=0.1, max_steps=50)
