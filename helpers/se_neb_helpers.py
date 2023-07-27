import os

from ase.build import sort
from ase.io import read, write

from helpers.generic_helpers import get_int_dirs, get_int_dirs_indices, need_sort
from helpers.generic_helpers import time_to_str, log_generic, need_sort
import os
from os.path import join as opj
from os.path import exists as ope
import time

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
