from ase.io import read, write
from os import mkdir as mkdir
from os.path import join as opj, exists as ope, basename
from time import time
import numpy as np

from helpers.generic_helpers import get_int_dirs, get_atoms, get_bond_str, get_atom_str, log_and_abort, \
    copy_state_files, remove_dir_recursive, time_to_str, need_sort, log_def, get_nrg, get_poscar_atoms
from helpers.geom_helpers import get_bond_length, get_atoms_prep_follow
from helpers.schedule_helpers import read_instructions_prep_input


def get_nrgs(work):
    int_dirs = get_int_dirs(work)
    fs = []
    for path in int_dirs:
        fs.append(get_nrg(path))
    return fs

def is_max(nrgs, i):
    return (nrgs[i + 1] < nrgs[i]) and (nrgs[i - 1] < nrgs[i])


def has_max(nrgs):
    for i in range(len(nrgs) - 2):
        if is_max(nrgs, i + 1):
            return True
    return False

def total_elapsed_str(start1, neb_time, scan_time):
    total_time = time() - start1
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

# def neb_optimizer(neb, neb_dir, opter, opt_alpha=150):
#     traj = opj(neb_dir, "opt.traj")
#     log = opj(neb_dir, "opt.log")
#     restart = opj(neb_dir, "hessian.pckl")
#     dyn = opter(neb, restart=restart, a=(opt_alpha / 70) * 0.1)
#     return dyn


def check_poscar(work_dir, log_fn):
    if need_sort(work_dir):
        atoms = get_poscar_atoms(work_dir, log_fn)
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


def safe_mode_check(scan_path, scan_steps_int, atom_pair_int_list, log_fn=log_def):
    def bond_length(step_idx):
        return get_bond_length(get_atoms(opj(scan_path, str(step_idx)), [False, False, False],
                                         restart_bool=True, log_fn=log_fn),
                               atom_pair_int_list)
    dstart = bond_length(0)
    dend = bond_length(scan_steps_int)
    dmax = dend - dstart
    sign = 1
    if dmax < 0:
        sign = -1
    dont_include = []
    for j in range(scan_steps_int - 1): # -1 so we don't accidentally exclude final optimization
        if sign*(bond_length(j) - dstart) > dmax:
            dont_include.append(j)
    include = []
    for j in range(scan_steps_int):
        if not j in dont_include:
            include.append(j)
    return include


def count_scan_steps(work_dir):
    scan_ints = []
    fname = opj(work_dir, "schedule")
    with open(fname, "r") as f:
        for line in f:
            if ":" in line:
                key = line.split(":")[0]
                if (not "#" in key) and (not "neb" in key):
                    scan_ints.append(int(key.strip()))
    return max(scan_ints) + 1


def _prep_input_bond(step_idx, atoms, prev_2_out, atom_pair, step_val, guess_type, step_dir,
                     val_target=False, log_func=log_def, carry_dict=None):
    print_str = ""
    prev_length = get_bond_length(atoms, atom_pair)
    log_func(f"Atom pair {get_bond_str(atoms, atom_pair[0], atom_pair[1])} previously at {prev_length:.5f}")
    if val_target:
        target_length = step_val
        step_length = target_length - prev_length
    else:
        target_length = prev_length + step_val
        step_length = step_val
    log_func(f"Creating structure with {get_bond_str(atoms, atom_pair[0], atom_pair[1])} at {target_length:.5f}")
    if (step_idx <= 1) and guess_type == 3:
        guess_type = 2
    if guess_type == 3:
        print_str += " atom momentum followed"
        atoms = get_atoms_prep_follow(atoms, prev_2_out, atom_pair, target_length)
    else:
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        dir_vec *= step_length / np.linalg.norm(dir_vec)
        if guess_type < 2:
            print_str += f" only {get_atom_str(atoms, atom_pair[guess_type])} moved"
            atoms.positions[atom_pair[guess_type]] += dir_vec
            if not carry_dict is None:
                if atom_pair[guess_type] in list(carry_dict.keys()):
                    for cidx in carry_dict[atom_pair[guess_type]]:
                        if not cidx in atom_pair:
                            atoms.positions[cidx] += dir_vec
        elif guess_type == 2:
            print_str += f" only {get_atom_str(atoms, atom_pair[0])} and {get_atom_str(atoms, atom_pair[1])} moved equidistantly"
            dir_vec *= 0.5
            atoms.positions[atom_pair[1]] += dir_vec
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
            if not carry_dict is None:
                if atom_pair[0] in list(carry_dict.keys()):
                    for cidx in carry_dict[atom_pair[0]]:
                        atoms.positions[cidx] += (-1) * dir_vec
                if atom_pair[1] in list(carry_dict.keys()):
                    for cidx in carry_dict[atom_pair[1]]:
                        atoms.positions[cidx] += dir_vec
    write(opj(step_dir, "POSCAR"), atoms, format="vasp")
    return print_str


def _prep_input(step_idx, schedule, step_dir, scan_dir, work_dir, carry_dict = None, log_fn=log_def):
    step_atoms, step_val, guess_type, target_bool = read_instructions_prep_input(schedule[str(step_idx)])
    step_prev_1_dir = opj(scan_dir, str(step_idx-1))
    step_prev_2_dir = opj(scan_dir, str(step_idx - 2))
    if step_idx == 0:
        atoms = get_atoms(work_dir, [False, False, False], False, log_fn)
    else:
        atoms = get_atoms(step_prev_1_dir, [False, False, False], True, log_fn)
    prev_2_out = opj(step_prev_2_dir, "CONTCAR")
    print_str = f"Prepared structure for step {step_idx} with"
    if len(step_atoms) == 2:
        print_str += _prep_input_bond(step_idx, atoms, prev_2_out, step_atoms, step_val, guess_type, step_dir,
                                      log_func=log_fn, val_target=target_bool, carry_dict = carry_dict)
        print_str += f" written to {opj(step_dir, 'POSCAR')}\n"
        print_str += "\t If you are restarting this step, the CONTCAR will be read, so the previous step does nothing \n"
        log_fn(print_str)
    else:
        log_and_abort("Non-bond scanning not yet implemented", log_fn=log_fn)


def setup_scan_dir(work_path, scan_path, restart_at_idx, pbc_bool_list, log_fn=log_def):
    dir0 = opj(scan_path, "0")
    if not ope(scan_path):
        log_fn("Creating scan directory")
        mkdir(scan_path)
    if not ope(dir0):
        log_fn(f"Setting up directory for step 0 (this is special for step 0 - please congratulate him)")
        mkdir(dir0)
        copy_state_files(work_path, dir0)
        atoms_obj = get_poscar_atoms(work_path, log_fn)
        atoms_obj = get_atoms(work_path, pbc_bool_list, restart_bool=True, log_fn=log_fn)
        write(opj(dir0, "POSCAR"), atoms_obj, format="vasp")
    log_fn("Checking for scan steps to be overwritten")
    int_dirs = get_int_dirs(work_path)
    for dir_path in int_dirs:
        idx = basename(dir_path)
        if idx > restart_at_idx:
            log_fn(f"Step {idx} comes after requested restart index {restart_at_idx}")
            remove_dir_recursive(dir_path)


