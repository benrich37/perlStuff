import os
from ase.io import read, write
import numpy as np
from generic_helpers import log_generic, atom_str


def get_start_dist(work_dir, atom_pair):
    atoms = read(work_dir + "0/CONTCAR")
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    return np.linalg.norm(dir_vec)

def read_neb_scan_inputs():
    """
    nImages: 10
    restart: True
    initial: POSCAR_start
    final: POSCAR_end
    work: /pscratch/sd/b/beri9208/1nPt1H_NEB/calcs/surfs/H2_H2O_start/No_bias/scan_bond_test/
    k: 0.2
    neb_method: spline
    interp_method: linear
    fix_pair: 0, 5
    fmax: 0.03
    """
    lookline = None
    restart_idx = 0
    max_steps = 100
    fmax = 0.05
    work_dir = None
    follow = False
    debug = False
    with open("neb_scan_input", "r") as f:
        for line in f:
            if not "#" in line:
                key = line.lower().split(":")[0]
                val = line.lower().rstrip("\n").split(":")[1]
                if "scan" in key:
                    lookline = val.split(",")
                if "restart" in key:
                    restart_idx = int(val.strip())
                if "debug" in key:
                    restart_bool_str = val
                    debug = "true" in restart_bool_str.lower()
                if "work" in key:
                    work_dir = val.strip()
                if "follow" in key:
                    follow = "true" in val
                if "max" in key:
                    if "steps" in key:
                        max_steps = int(val.strip())
                    elif ("force" in key) or ("fmax" in key):
                        fmax = float(val.strip())
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    if work_dir is None:
        work_dir = os.getcwd()
    if work_dir[-1] != "/":
        work_dir += "/"
    return atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, debug, max_steps, fmax
def read_scan_inputs():
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
    with open("scan_input", "r") as f:
        for line in f:
            if "scan" in line.lower().split(":")[0]:
                lookline = line.rstrip("\n").split(":")[1].split(",")
            if "restart" in line.lower().split(":")[0]:
                restart_idx = int(line.rstrip("\n").split(":")[1])
            if "work" in line.lower().split(":")[0]:
                work_dir = line.rstrip("\n").split(":")[1]
            if "follow" in line.lower().split(":")[0]:
                follow = "true" in line.lower().split(":")[1]
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    if work_dir is None:
        work_dir = os.getcwd()
    if work_dir[-1] != "/":
        work_dir += "/"
    return atom_pair, scan_steps, step_length, restart_idx, work_dir, follow


def _scan_log(log_str, work, print_bool=False):
    log_generic(log_str, work, "bond_scan", print_bool)


def _prep_input(step_idx, atom_pair, step_length, start_length, follow, log_func, work_dir, step_type):
    if not work_dir == os.getcwd():
        os.chdir(work_dir)
    print_str = f"Prepared structure for step {step_idx} with"
    target_length = start_length + (step_idx*step_length)
    atoms = read(str(step_idx - 1) + "/CONTCAR", format="vasp")
    if step_idx <= 1:
        follow = False
    if follow:
        print_str += " atom momentum followed"
        atoms_prev = read(str(step_idx - 2) + "/CONTCAR", format="vasp")
        dir_vecs = []
        for i in range(len(atoms.positions)):
            dir_vecs.append(atoms.positions[i] - atoms_prev.positions[i])
        for i in range(len(dir_vecs)):
            atoms.positions[i] += dir_vecs[i]
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        cur_length = np.linalg.norm(dir_vec)
        should_be_0 = target_length - cur_length
        if not np.isclose(should_be_0, 0.0):
            atoms.positions[atom_pair[1]] += dir_vec*(should_be_0)/np.linalg.norm(dir_vec)
    else:
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        dir_vec *= step_length / np.linalg.norm(dir_vec)
        if step_type == 0:
            print_str += f" only {atom_str(atoms, atom_pair[1])} moved"
            atoms.positions[atom_pair[1]] += dir_vec
        elif step_type == 1:
            print_str += f" only {atom_str(atoms, atom_pair[0])} and {atom_str(atoms, atom_pair[1])} moved equidistantly"
            dir_vec *= 0.5
            atoms.positions[atom_pair[1]] += dir_vec
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
        elif step_type == 2:
            print_str += f" only {atom_str(atoms, atom_pair[0])} moved"
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
    write(str(step_idx) + "/POSCAR", atoms, format="vasp")
    log_func(print_str)

