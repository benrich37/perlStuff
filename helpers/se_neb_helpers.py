
from ase.build import sort
from ase.io import read, write
from os import mkdir as mkdir
from os.path import join as opj, exists as ope, basename
from time import time
import numpy as np

from helpers.generic_helpers import get_int_dirs, get_int_dirs_indices, get_atoms, bond_str, atom_str, log_and_abort, \
    copy_state_files, remove_dir_recursive, time_to_str,  need_sort, log_def
from helpers.geom_helpers import get_bond_length, get_atoms_prep_follow


def get_f(path):
    with open(opj(path, "Ecomponents")) as f:
        for line in f:
            if "F =" in line:
                return float(line.strip().split("=")[1])

def get_fs(work):
    int_dirs = get_int_dirs(work)
    fs = []
    for path in int_dirs:
        fs.append(get_f(path))
    return fs

def has_max(fs):
    for i in range(len(fs) - 2):
        if fs[i + 2] > fs[i + 1]:
            if fs[i] < fs[i + 1]:
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
            info = section_split[1:]
            if "bs" in command:
                scan_pair = [int(int(info[0]) - 1), int(int(info[1]) - 1)]
                step_val = info[2]
                target_bool = not (("+" in step_val) or ("-" in step_val))
                step_val = float(step_val)
                guess_type = int(info[3])
            elif "j" in command:
                j_steps = int(info[0])
            elif "f" in command:
                nAtoms = len(info)
                freeze_tuple = []
                for i in range(nAtoms):
                    freeze_tuple.append(int(int(info[i].strip()) - 1))
                constraint_tuples.append(tuple(freeze_tuple))
    return scan_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples


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
    scan_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples = read_schedule_line_helper(val)
    return idx, scan_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples


step_atoms_key = "step_atoms"
step_size_key = "step_size"
guess_type_key = "guess_type"
j_steps_key = "jdftx_steps"
freeze_list_key = "freeze_list"
target_bool_key = "target"
energy_key = "nrg"
properties_key = "props"


def write_step_to_schedule_dict(schedule, idx, scan_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples):
    schedule[str(idx)] = {}
    schedule[str(idx)][step_atoms_key] = scan_pair
    schedule[str(idx)][step_size_key] = step_val
    schedule[str(idx)][target_bool_key] = target_bool
    schedule[str(idx)][guess_type_key] = guess_type
    schedule[str(idx)][j_steps_key] = j_steps
    schedule[str(idx)][freeze_list_key] = constraint_tuples
    schedule[str(idx)][energy_key] = None
    schedule[str(idx)][properties_key] = None


def count_scan_steps_from_schedule(schedule):
    scan_steps = []
    for key in schedule.keys():
        try:
            scan_int = int(key)
            scan_steps.append(scan_int)
        except:
            pass
    return max(scan_steps) + 1


def get_nrg_comment_str(schedule, i):
    nrg = schedule[i][energy_key]
    return f"{nrg}"

def get_prop_comment_str(prop):
    prop_str = ""
    nMems = len(prop) - 1
    for i in range(nMems):
        if i > 0:
            prop_str += ", "
        prop_str += f"{prop[i]}"
    prop_str += f" = {prop[-1]}"
    return prop_str


def get_props_comment_str(schedule, i):
    return_str = ""
    props = schedule[i][properties_key]
    for prop in props:
        return_str += get_prop_comment_str(prop)
    return return_str


def get_step_results_comment_str(schedule, i):
    comment_str = f"# {i}:"
    comment_str += get_nrg_comment_str(schedule, i)
    comment_str += get_props_comment_str(schedule, i)
    comment_str += "\n"
    return comment_str

def get_step_results_insert_index(i, contents):
    for idx, line in enumerate(contents):
        if ":" in line:
            if str(i) == line.split(":")[0].strip():
                return idx
    raise ValueError(f"Could not find appropriate insert index for step {i}")



def append_step_results_to_contents(schedule, i, contents):
    comment_str = get_step_results_comment_str(schedule, i)
    idx = get_step_results_insert_index(i, contents)
    contents.insert(idx, comment_str)
    return contents



def append_comments_to_contents(schedule, contents):
    nSteps = count_scan_steps_from_schedule(schedule)
    for i in range(nSteps):
        nrg = schedule[str(i)][energy_key]
        if not nrg is None:
            contents = append_step_results_to_contents(schedule, i, contents)

def append_results_as_comments(schedule, work_dir):
    fname = opj(work_dir, "schedule")
    with open(fname, "r") as f:
        contents = f.readlines()
    append_comments_to_contents(schedule, contents)
    with open(fname, "w") as f:
        contents = "".join(contents)
        f.write(contents)



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
                        idx, scan_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples = read_schedule_step_line(line)
                        write_step_to_schedule_dict(schedule, idx, scan_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples)
    return schedule


def get_schedule_step_str_commands(atom_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples):
    dump_str = f"bs, {int(atom_pair[0] + 1)}, {int(atom_pair[1] + 1)}, "
    if not target_bool:
        if step_val >= 0:
            dump_str += "+"
    dump_str += f"{float(step_val)}, {int(guess_type)}|"
    dump_str += f"j, {int(j_steps)}|"
    for c in constraint_tuples:
        dump_str += "f, "
        for atom in c:
            dump_str += f"{int(atom + 1)}, "
        dump_str += "|"
    return dump_str


def get_schedule_step_str(i, atom_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples):
    dump_str = f"{i}: "
    dump_str += get_schedule_step_str_commands(atom_pair, step_val, target_bool, guess_type, j_steps, constraint_tuples)
    dump_str += "\n"
    return dump_str






def autofill_schedule(step_atoms, scan_steps, step_size, guess_type, j_steps, constraint_tuples,
                      relax_start, relax_end, neb_steps, k, neb_method):
    schedule = {}
    for idx in range(scan_steps):
        use_step = step_size
        if idx == 0:
            use_step = 0.0
        if ((idx == 0) and relax_start) or ((idx == scan_steps) and relax_end):
            write_step_to_schedule_dict(schedule, idx, step_atoms, use_step, False, guess_type, j_steps, [])
        else:
            write_step_to_schedule_dict(schedule, idx, step_atoms, use_step, False, guess_type, j_steps, constraint_tuples)
    write_neb_to_schedule_dict(schedule, neb_steps, k, neb_method, list(range(scan_steps)))
    return schedule


def write_autofill_schedule(atom_pair, scan_steps, step_length, guess_type, j_steps, constraint_tuples, relax_start,
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
    step_val = schedule_dict_val[step_size_key]
    guess_type = schedule_dict_val[guess_type_key]
    target_bool = schedule_dict_val[target_bool_key]
    nAtoms = len(step_atoms)
    if nAtoms == 2:
        dump_str += "bs, "
    else:
        raise ValueError("Scans for angles not yet implemented")
    for idx in step_atoms:
        dump_str += f"{idx + 1}, "
    if target_bool:
        dump_str += f"{step_val}, "
    else:
        if step_val >= 0:
            dump_str += "+"
        dump_str += f"{step_val}, "
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


def get_step_list(schedule, restart_at):
    step_list = []
    for key in schedule.keys():
        try:
            step = int(key)
            if step >= restart_at:
                step_list.append(step)
        except:
            pass
    return step_list


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
    return max(scan_ints) - 1


def _prep_input_bond(step_idx, atoms, prev_2_out, atom_pair, step_val, guess_type, step_dir,
                     val_target=False, log_func=log_def):
    print_str = ""
    prev_length = get_bond_length(atoms, atom_pair)
    log_func(f"Atom pair {bond_str(atoms, atom_pair[0], atom_pair[1])} previously at {prev_length}")
    if val_target:
        target_length = step_val
        step_length = target_length - prev_length
    else:
        target_length = prev_length + step_val
        step_length = step_val
    log_func(f"Creating structure with {bond_str(atoms, atom_pair[0], atom_pair[1])} at {target_length}")
    if (step_idx <= 1) and guess_type == 3:
        guess_type = 2
    if guess_type == 3:
        print_str += " atom momentum followed"
        atoms = get_atoms_prep_follow(atoms, prev_2_out, atom_pair, target_length)
    else:
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        dir_vec *= step_length / np.linalg.norm(dir_vec)
        if guess_type == 0:
            print_str += f" only {atom_str(atoms, atom_pair[0])} moved"
            atoms.positions[atom_pair[1]] += dir_vec
        elif guess_type == 1:
            print_str += f" only {atom_str(atoms, atom_pair[0])} moved"
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
        elif guess_type == 2:
            print_str += f" only {atom_str(atoms, atom_pair[0])} and {atom_str(atoms, atom_pair[1])} moved equidistantly"
            dir_vec *= 0.5
            atoms.positions[atom_pair[1]] += dir_vec
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
    write(opj(step_dir, "POSCAR"), atoms, format="vasp")
    return print_str


def _prep_input(step_idx, schedule, step_dir, scan_dir, work_dir, log_fn=log_def):
    step_atoms, step_val, guess_type, target_bool = read_instructions_prep_input(schedule[str(step_idx)])
    step_prev_1_dir = opj(scan_dir, str(step_idx-1))
    step_prev_2_dir = opj(scan_dir, str(step_idx - 2))
    if step_idx == 0:
        prev_1_out = opj(work_dir, "POSCAR")
    else:
        prev_1_out = opj(step_prev_1_dir, "CONTCAR")
    atoms = read(prev_1_out, format="vasp")
    prev_2_out = opj(step_prev_2_dir, "CONTCAR")
    print_str = f"Prepared structure for step {step_idx} with"
    if len(step_atoms) == 2:
        print_str += _prep_input_bond(step_idx, atoms, prev_2_out, step_atoms, step_val, guess_type, step_dir,
                                      log_func=log_fn, val_target=target_bool)
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
        atoms_obj = get_atoms(work_path, pbc_bool_list, restart_bool=True, log_fn=log_fn)
        write(opj(dir0, "POSCAR"), atoms_obj, format="vasp")
    log_fn("Checking for scan steps to be overwritten")
    int_dirs = get_int_dirs(work_path)
    for dir_path in int_dirs:
        idx = basename(dir_path)
        if idx > restart_at_idx:
            log_fn(f"Step {idx} comes after requested restart index {restart_at_idx}")
            remove_dir_recursive(dir_path)


def read_instructions_prep_input(instructions):
    step_atoms = instructions[step_atoms_key]
    step_size = instructions[step_size_key]
    guess_type = instructions[guess_type_key]
    target_bool = instructions[target_bool_key]
    return step_atoms, step_size, guess_type, target_bool
