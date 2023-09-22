import os
from os.path import join as opj
from os.path import exists as ope
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from JDFTx import JDFTx
from helpers.generic_helpers import copy_state_files, get_cmds_dict, get_inputs_list, fix_work_dir, optimizer, dump_template_input
from helpers.generic_helpers import add_bond_constraints, read_pbc_val, write_contcar
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.scan_bond_helpers import _scan_log, _prep_input


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



bond_scan_template = ["scan: 3, 5, 10, 0.23",
                   "restart: 3",
                   "max_steps: 100",
                   "fmax: 0.05",
                   "follow: False",
                   "pbc: True, true, false"]

def read_scan_inputs(fname = "scan_input"):
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
    gpu = True
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
        if "gpu" in key:
            gpu = "true" in val.lower()
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    work_dir = fix_work_dir(work_dir)
    return atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, pbc, gpu

def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write("Done")


def run_step(step_dir, fix_pair, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(step_dir, "POSCAR"), format="vasp")
    atoms.pbc = [True, True, False]
    add_bond_constraints(atoms, fix_pair)
    scan_log("creating calculator")
    calculator = get_calc(step_dir)
    scan_log("setting calculator")
    atoms.set_calculator(calculator)
    scan_log("printing atoms")
    scan_log(atoms)
    scan_log("setting optimizer")
    dyn = optimizer(atoms, step_dir, FIRE)
    traj = Trajectory(step_dir +'opt.traj', 'w', atoms, properties=['energy', 'forces'])
    scan_log("attaching trajectory")
    dyn.attach(traj.write, interval=1)
    dyn.attach(lambda: write_contcar(atoms, step_dir), interval=1)
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        finished(step_dir)
    except Exception as e:
        scan_log("couldnt run??")
        scan_log(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)


if __name__ == '__main__':
    atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, pbc, gpu = read_scan_inputs()
    scan_log = lambda s: _scan_log(s, work_dir)
    exe_cmd = get_exe_cmd(gpu, scan_log)
    cmds = get_cmds_dict(work_dir)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, log_fn=scan_log)
    prep_input = lambda s, a, l: _prep_input(s, a, l, follow, scan_log, work_dir, 1)
    os.chdir(work_dir)
    if (not os.path.exists("./0")) or (not os.path.isdir("./0")):
        os.mkdir("./0")
    copy_state_files("./", "./0")
    for i in list(range(scan_steps))[restart_idx:]:
        if (not os.path.exists(f"./{str(i)}")) or (not os.path.isdir(f"./{str(i)}")):
            os.mkdir(f"./{str(i)}")
        if i > 0:
            copy_state_files(f"./{str(i - 1)}", f"./{str(i)}")
        if (i > 1):
            prep_input(i, atom_pair, step_length)
        run_step(opj(work_dir, str(i)), atom_pair, fmax=0.1, max_steps=50)
