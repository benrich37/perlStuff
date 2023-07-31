import os

from ase.build import sort
from ase.io import read, write

from helpers.generic_helpers import get_int_dirs, get_int_dirs_indices, need_sort
from helpers.generic_helpers import time_to_str, log_generic, need_sort, log_def
import os
from os.path import join as opj
from os.path import exists as ope
import time
import numpy as np


def get_f(path):
    with open(os.path.join(path, "Ecomponents")) as f:
        for line in f:
            if "F =" in line:
                return float(line.strip().split("=")[1])

def get_fs(work):
    int_dirs = get_int_dirs(work)
    indices = get_int_dirs_indices(int_dirs)
    fs = []
    for i in indices:
        fs.append(get_f(os.path.join(work, int_dirs[i])))
    return fs

def has_max(fs):
    for i in range(len(fs) - 2):
        if fs[i + 2] > fs[i + 1]:
            if fs[i] < fs[i + 1]:
                return True
    return False

def total_elapsed_str(start1, neb_time, scan_time):
    total_time = time.time() - start1
    print_str = f"Total time: " + time_to_str(total_time) + "\n"
    print_str += f"Scan opt time: {time_to_str(scan_time)} ({(scan_time / total_time):.{2}g}%)\n"
    print_str += f"NEB opt time: {time_to_str(neb_time)} ({(neb_time / total_time):.{2}g}%)\n"
    return print_str


def log_total_elapsed(start1, neb_time, scan_time, work):
    log_fname = opj(work, "time.log")
    if not ope(log_fname):
        with open(log_fname, "w") as f:
            f.write("Starting")
            f.close()
    with open(log_fname, "a") as f:
        f.write(total_elapsed_str(start1, neb_time, scan_time))

def neb_optimizer(neb, neb_dir, opter, opt_alpha=150):
    traj = opj(neb_dir, "opt.traj")
    log = opj(neb_dir, "opt.log")
    restart = opj(neb_dir, "hessian.pckl")
    dyn = opter(neb, trajectory=traj, logfile=log, restart=restart, a=(opt_alpha / 70) * 0.1)
    return dyn


def check_poscar(work_dir, log_fn):
    if need_sort(work_dir):
        atoms = sort(read("POSCAR", format="vasp"))
        write("POSCAR_sorted", atoms, format="vasp")
        log_fn("Unsorted POSCAR - saved sorted POCSAR to POSCAR_sorted - please update atom indices for scan accordingly", work_dir)
        raise ValueError("Unsorted POSCAR - saved sorted POCSAR to POSCAR_sorted - please update atom indices for scan accordingly")

def fix_step_size(starting_length, target_length, nSteps, log_fn = log_def):
    dLength = target_length - starting_length
    fixed = dLength/(nSteps - 1) # Step 0 is a step, so subtract 1 from nSteps
    log_fn(f"Updating step size to {fixed}")
    return fixed

def get_index_map(atoms):
    index_map = {}
    symbols = atoms.get_chemical_symbols()
    for i, atom in enumerate(symbols):
        if not atom in index_map:
            index_map[atom] = []
        index_map[atom].append(i)
    return index_map

def get_sorted_positions(atoms_old, atoms_new):
    map1 = get_index_map(atoms_old)
    map2 = get_index_map(atoms_new)
    posns_new = atoms_new.positions
    posns_sorted = np.zeros(np.shape(posns_new))
    for atom in list(map1.keys()):
        assert(len(map1[atom]) == len(map2[atom]))
        for i in range(len(map1[atom])):
            posns_sorted[map1[atom][i]] = posns_new[map2[atom][i]]
    return posns_sorted

def read_schedule_line(line):
    """ Example line:
    0: 54, 55, 0.1, 1 (#increase bond 54-55 by 0.1 with guess type #1 (move second atom))
    1: 54, 55, 0.1, 0 (#same but move first atom only for guess)
    2: 55, 58, 0.5, 2 (#Increase bond 55-58 by 0.5 moving both equidistant)
    3: 55, 58, 0.5, 3 (#Same but use momentum following)
    :return:
    """
    idx = int(line.split(":")[0])
    valsplit = line.split(":")[1].split(",")
    atom_pair = [int(valsplit[0] - 1), int(valsplit[1] - 1)]
    dx = float(valsplit[2])
    guess_idx = int(valsplit[3])
    return idx, atom_pair, dx, guess_idx

def read_schedule_file(fname):
    schedule = {}
    with open(fname, "r") as f:
        for line in f:
            step_idx, atom_pair, dx, guess_type = read_schedule_line(line)

