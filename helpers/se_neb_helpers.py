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


def read_schedule_line_helper(val):
    val_sections = val.strip().split("|")
    constraint_tuples = []
    for section in val_sections:
        if len(section) > 1:
            section_split = section.strip().split(",")
            command = section_split[0]
            if "bs" in command:
                scan_pair = [int(int(section_split[1]) - 1), int(int(section_split[2]) - 1)]
                dx = float(section_split[3])
                guess_type = int(section_split[4])
            elif "j" in command:
                j_steps = int(section_split[1])
            elif "f" in command:
                nAtoms = len(section_split) - 1
                freeze_tuple = []
                for i in range(nAtoms):
                    try:
                        freeze_tuple.append(int(int(section_split[i]) - 1))
                    except:
                        pass
                constraint_tuples.append(tuple(freeze_tuple))
    return scan_pair, dx, guess_type, j_steps, constraint_tuples


def get_schedule_neb_str(schedule_dict_val):
    extract_steps = schedule_dict_val[extract_steps_key]
    neb_steps =schedule_dict_val[neb_steps_key]
    k = schedule_dict_val[neb_k_key]
    neb_method = schedule_dict_val[neb_method_key]
    dump_str = f"{neb_key}: "
    dump_str += f"{neb_steps}, {k}, {neb_method} |"
    dump_str += f"{extract_steps[0]}"
    for i in range(len(extract_steps) - 1):
        dump_str += f", {extract_steps[i+1]}"
    return dump_str

def read_schedule_neb_line(line):
    groups = line.split(":")[1].rstrip("\n").strip().split("|")
    g1 = groups[0].split(",")
    g2 = groups[1].split(",")
    neb_steps = int(g1[0])
    k = float(g1[1])
    neb_method = str(g1[2])
    extract_steps = []
    for v in g2:
        try:
            extract_steps.append(int(v))
        except:
            pass
    return neb_steps, k, neb_method, extract_steps




def read_schedule_step_line(line):
    """ Example line:
    0: bs, 54, 55, 0.1, 1 | j, 100 | f, 54, 55 # (increase bond 54-55 by 0.1 with guess type #1 (move second atom), and run jdftx 100 steps before ASE)
    1: bs, 54, 55, 0.1, 0 | j, 0 | f, 55, 58 # (same but move first atom only for guess, run 0 jdft steps, and freeze bond 55-58 as well)
    2: bs, 55, 58, 0.5, 2 (#Increase bond 55-58 by 0.5 moving both equidistant)
    3: bs, 55, 58, 0.5, 3 (#Same but use momentum following)
    :return:
    """
    idx = int(line.split(":")[0])
    val = line.split(":")[1]
    scan_pair, dx, guess_type, j_steps, constraint_tuples = read_schedule_line_helper(val)
    return idx, scan_pair, dx, guess_type, j_steps, constraint_tuples


step_atoms_key = "step_atoms"
step_size_key = "step_size"
guess_type_key = "guess_type"
j_steps_key = "jdftx_steps"
freeze_list_key = "freeze_list"


def write_step_to_schedule_dict(schedule, idx, scan_pair, dx, guess_type, j_steps, constraint_tuples):
    schedule[str(idx)] = {}
    schedule[str(idx)][step_atoms_key] = scan_pair
    schedule[str(idx)][step_size_key] = dx
    schedule[str(idx)][guess_type_key] = guess_type
    schedule[str(idx)][j_steps_key] = j_steps
    schedule[str(idx)][freeze_list_key] = constraint_tuples


neb_key = "neb"
neb_steps_key = "neb_steps"
neb_k_key = "k"
neb_method_key = "neb_method"
extract_steps_key = "from"


def write_neb_to_schedule_dict(schedule, neb_steps, k, neb_method, extract_step_idcs):
    schedule[neb_key] = {}
    schedule[neb_key][neb_steps_key] = neb_steps
    schedule[neb_key][neb_k_key] = k
    schedule[neb_key][neb_method_key] = neb_method
    schedule[neb_key][extract_steps_key] = extract_step_idcs



def read_schedule_file(root_path):
    fname = opj(root_path, "schedule")
    schedule = {}
    with open(fname, "r") as f:
        for line in f:
            if ":" in line:
                if not "#" in line.split(":")[0]:
                    if line.split(":")[0] == neb_key:
                        neb_steps, k, neb_method, extract_steps = read_schedule_neb_line(line)
                        write_neb_to_schedule_dict(schedule, neb_steps, k, neb_method, extract_steps)
                    else:
                        idx, scan_pair, dx, guess_type, j_steps, constraint_tuples = read_schedule_step_line(line)
                        write_step_to_schedule_dict(schedule, idx, scan_pair, dx, guess_type, j_steps, constraint_tuples)
    return schedule


def get_schedule_step_str_commands(atom_pair, step_length, guess_type, j_steps, constraint_tuples):
    dump_str = f"bs, {int(atom_pair[0] + 1)}, {int(atom_pair[1] + 1)}, {float(step_length)}, {int(guess_type)}|"
    dump_str += f"j, {int(j_steps)}|"
    for c in constraint_tuples:
        dump_str += "f, "
        for atom in c:
            dump_str += f"{int(atom + 1)}, "
        dump_str += "|"
    return dump_str


def get_schedule_step_str(i, atom_pair, step_length, guess_type, j_steps, constraint_tuples):
    dump_str = f"{i}: "
    dump_str += get_schedule_step_str_commands(atom_pair, step_length, guess_type, j_steps, constraint_tuples)
    dump_str += "\n"
    return dump_str






def autofill_schedule(step_atoms, scan_steps, step_size, guess_type, j_steps, constraint_tuples,
                      relax_start, relax_end, neb_steps, k, neb_method):
    schedule = {}
    for idx in range(scan_steps):
        if ((idx == 0) and relax_start) or ((idx == scan_steps) and relax_end):
            write_step_to_schedule_dict(schedule, idx, step_atoms, step_size, guess_type, j_steps, [])
        else:
            write_step_to_schedule_dict(schedule, idx, step_atoms, step_size, guess_type, j_steps, constraint_tuples)
    write_neb_to_schedule_dict(schedule, neb_steps, k, neb_method, list(range(scan_steps)))
    return schedule


def write_auto_schedule(atom_pair, scan_steps, step_length, guess_type, j_steps, constraint_tuples, relax_start,
                        relax_end, neb_steps, k, neb_method, work_dir):
    fname = opj(work_dir, "schedule")
    schedule = autofill_schedule(atom_pair, scan_steps, step_length, guess_type, j_steps, constraint_tuples,
                                 relax_start, relax_end, neb_steps, k, neb_method)
    dump_str = get_schedule_dict_str(schedule)
    with open(fname, "w") as f:
        f.write(dump_str)

def write_step_command(schedule_dict_val):
    dump_str = ""
    step_atoms = schedule_dict_val[step_atoms_key]
    step_size = schedule_dict_val[step_size_key]
    guess_type = schedule_dict_val[guess_type_key]
    nAtoms = len(step_atoms)
    if nAtoms == 2:
        dump_str += "bs, "
    else:
        raise ValueError("Scans for angles not yet implemented")
    for idx in step_atoms:
        dump_str += f"{idx + 1}, "
    dump_str += f"{step_size}, "
    dump_str += f"{guess_type}|"
    return dump_str

def write_jdftx_steps(schedule_dict_val):
    dump_str = "j, "
    j_steps = schedule_dict_val[j_steps_key]
    dump_str += f"{j_steps}|"
    return dump_str

def write_freeze_list(schedule_dict_val):
    dump_str = ""
    freeze_list = schedule_dict_val[freeze_list_key]
    for group in freeze_list:
        dump_str += "f"
        for idx in group:
            dump_str += f", {idx + 1}"
        dump_str += "|"
    return dump_str


def get_schedule_dict_str(schedule_dict):
    dump_str = ""
    for key in schedule_dict.keys():
        if not (key == neb_key):
            dump_str += str(key) + ": "
            dump_str += write_step_command(schedule_dict[key])
            dump_str += write_jdftx_steps(schedule_dict[key])
            dump_str += write_freeze_list(schedule_dict[key])
            dump_str += "\n"
        else:
            dump_str += get_schedule_neb_str(schedule_dict[key])
            dump_str += "\n"
    return dump_str

def write_schedule_dict(schedule_dict, work_dir):
    fname = opj(work_dir, "schedule")
    with open(fname, "w") as f:
        f.write(get_schedule_dict_str(schedule_dict))

