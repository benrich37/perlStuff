import os
import shutil
from ase.io import read
import time
import numpy as np
from datetime import datetime
from ase.constraints import FixBondLength
from os.path import join as opj
from os.path import exists as ope
from ase.io import read, write


gbrv_15_ref = [
    "sn f ca ta sc cd sb mg b se ga os ir li si co cr pt cu i pd br k as h mn cs rb ge bi ag fe tc hf ba ru al hg mo y re s tl te ti be p zn sr n rh au hf nb c w ni cl la in v pb zr o ",
    "14. 7. 10. 13. 11. 12. 15. 10. 3. 6. 19. 16. 15. 3. 4. 17. 14. 16. 19. 7. 16. 7. 9. 5. 1. 15. 9. 9. 14. 15. 19. 16. 15. 12. 10. 16. 3. 12. 14. 11. 15. 6. 13. 6. 12. 4. 5. 20. 10. 5. 15. 11. 12. 13. 4. 14. 18. 7. 11. 13. 13. 14. 12. 6. "
]

valence_electrons = {
        'h': 1, 'he': 2,
        'li': 1, 'be': 2, 'b': 3, 'c': 4, 'n': 5, 'o': 6, 'f': 7, 'ne': 8,
        'na': 1, 'mg': 2, 'al': 3, 'si': 4, 'p': 5, 's': 6, 'cl': 7, 'ar': 8,
        'k': 1, 'ca': 2, 'sc': 2, 'ti': 2, 'v': 2, 'cr': 1, 'mn': 2, 'fe': 2, 'co': 2, 'ni': 2, 'cu': 1, 'zn': 2,
        'ga': 3, 'ge': 4, 'as': 5, 'se': 6, 'br': 7, 'kr': 8,
        'rb': 1, 'sr': 2, 'y': 2, 'zr': 2, 'nb': 1, 'mo': 1, 'tc': 2, 'ru': 2, 'rh': 1, 'pd': 0, 'ag': 1, 'cd': 2,
        'in': 3, 'sn': 4, 'sb': 5, 'te': 6, 'i': 7, 'xe': 8,
        'cs': 1, 'ba': 2, 'la': 2, 'ce': 2, 'pr': 2, 'nd': 2, 'pm': 2, 'sm': 2, 'eu': 2, 'gd': 3, 'tb': 3, 'dy': 3,
        'ho': 3, 'er': 3, 'tm': 2, 'yb': 2, 'lu': 2, 'hf': 2, 'ta': 2, 'w': 2, 're': 2, 'os': 2, 'ir': 2, 'pt': 2,
        'au': 1, 'hg': 2, 'tl': 3, 'pb': 4, 'bi': 5, 'po': 6, 'at': 7, 'rn': 8,
    }

def copy_files(src_dir, tgt_dir):
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, tgt_dir)

def get_int_dirs_indices(int_dirs):
    ints = []
    for dirr in int_dirs:
        ints.append(int(dirr.split("/")[-1]))
    return np.array(ints).argsort()


def get_int_dirs(dir_path):
    int_dir_list = []
    for name in os.listdir(dir_path):
        full_path = os.path.join(dir_path, name)
        if os.path.isdir(full_path):
            try:
                int(name)
                int_dir_list.append(full_path)
            except ValueError:
                continue
    return int_dir_list

def insert_el(filename):
    """
    Inserts elements line in correct position for Vasp 5? Good for
    nebmovie.pl script in VTST-tools package
    Args:
        filename: name of file to add elements line
    """
    with open(filename, 'r') as f:
        file = f.read()
    contents = file.split('\n')
    ele_line = contents[0]
    if contents[5].split() != ele_line.split():
        contents.insert(5, ele_line)
    with open(filename, 'w') as f:
        f.write('\n'.join(contents))

def read_inputs(work_dir, ref_struct=None):
    inpfname = opj(work_dir, "inputs")
    if os.path.exists("inputs"):
        ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump", "initial-state",
                  "coulomb-interaction", "coulomb-truncation-embed", "lattice-type", "opt", "max_steps", "fmax",
                  "optimizer", "pseudos", "logfile", "restart", "econv", "safe-mode"]
        input_cmds = {"dump": "End State"}
        with open(inpfname) as f:
            for i, line in enumerate(f):
                if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                    skip = False
                    for ig in ignore:
                        if ig in line:
                            skip = True
                    if "#" in line:
                        skip = True
                    if "ASE" in line:
                        break
                    if not skip:
                        cmd = line[:line.index(" ")]
                        rest = line.rstrip("\n")[line.index(" ") + 1:]
                        if not cmd in ignore:
                            input_cmds[cmd] = rest
        do_n_bands = False
        if "elec-n-bands" in input_cmds.keys():
            if input_cmds["elec-n-bands"] == "*":
                do_n_bands = True
        else:
            do_n_bands = True
        if do_n_bands:
            ref_paths = [opj(work_dir, ref_struct), opj(work_dir, "CONTCAR"), opj(work_dir, "POSCAR")]
            for p in ref_paths:
                if ope(p):
                    input_cmds["elec-n-bands"] = str(get_nbands(p))
                    break
        return input_cmds
    else:
        return None

def get_nbands(poscar_fname):
    atoms = read(poscar_fname)
    count_dict = {}
    for a in atoms.get_chemical_symbols():
        if a.lower() not in count_dict.keys():
            count_dict[a.lower()] = 0
        count_dict[a.lower()] += 1
    nval = 0
    for a in count_dict.keys():
        if a in gbrv_15_ref[0].split(" "):
            idx = gbrv_15_ref[0].split(" ").index(a)
            val = (gbrv_15_ref[1].split(". "))[idx]
            count = count_dict[a]
            nval += int(val) * int(count)
        else:
            nval += int(valence_electrons[a]) * int(count_dict[a])
    return max([int(nval / 2) + 10, int((nval / 2) * 1.2)])


def dup_cmds(infile):
    lattice_line = None
    infile_cmds = {}
    infile_cmds["dump"] = "End State"
    ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump-name", "initial-state",
              "coulomb-interaction", "coulomb-truncation-embed"]
    with open(infile) as f:
        for i, line in enumerate(f):
            if "lattice " in line:
                lattice_line = i
            if not lattice_line is None:
                if i > lattice_line + 3:
                    if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                        skip = False
                        for ig in ignore:
                            if ig in line:
                                skip = True
                            elif line[:4] == "ion ":
                                skip = True
                        if not skip:
                            cmd = line[:line.index(" ")]
                            rest = line.rstrip("\n")[line.index(" ") + 1:]
                            if not cmd in ignore:
                                if not cmd == "dump":
                                    infile_cmds[cmd] = rest
    return infile_cmds

def copy_rel_files(src, dest):
    state_files = ["wfns", "eigenvals", "fillings", "fluidState","force","CONTCAR","POSCAR", "hessian.pckl","in"]
    for f in state_files:
        if os.path.exists(os.path.join(src, f)):
            print(f"copying {f} from {src} to {dest}")
            shutil.copy(os.path.join(src, f), dest)

def remove_restart_files(dir):
    state_files = ["wfns", "eigenvals", "fillings", "fluidState", "force", "hessian.pckl"]
    for f in state_files:
        if os.path.exists(os.path.join(dir, f)):
            print(f"removing {f} from {dir}")
            os.remove(os.path.join(dir, f))


def time_to_str(t):
    if t < 60:
        print_str =  f"{t:.{3}g} sec"
    elif t < 3600:
        print_str =  f"{t/60.:.{3}g} min"
    else:
        print_str = f"{t /3600.:.{3}g} hr"
    return print_str


def atom_str(atoms, index):
    return f"{atoms.get_chemical_symbols()[index]}({index})"

def need_sort(root):
    atoms = read(os.path.join(root, "POSCAR"), format="vasp")
    ats = []
    dones = []
    for a in atoms.get_chemical_symbols():
        if not a in ats:
            ats.append(ats)
        elif a in dones:
            return True
        for at in ats:
            if not at in dones:
                if at != a:
                    dones.append(at)
    return False

def log_generic(message, work, calc_type, print_bool):
    print(message)
    if not "\n" in message:
        message = message + "\n"
        print(message)
    prefix = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    message = prefix + message
    log_fname = os.path.join(work, calc_type + ".log")
    if not os.path.exists(log_fname):
        with open(log_fname, "w") as f:
            f.write(prefix + "Starting\n")
            f.close()
    with open(log_fname, "a") as f:
        f.write(message)
        f.close()
    if print_bool:
        print(message)

def get_cmds(work_dir, ref_struct = None):
    os.chdir(work_dir)
    if not ope(opj(work_dir, "inputs")):
        return dup_cmds(opj(work_dir, "in"))
    else:
        return read_inputs(work_dir, ref_struct=ref_struct)



def read_line_generic(line):
    key = line.lower().split(":")[0]
    val = line.rstrip("\n").split(":")[1]
    if not "#" in key:
        if "#" in val:
            val = val[:val.index("#")]
        return key, val
    else:
        return None, None


def remove_dir_recursive(path):
    for root, dirs, files in os.walk(path, topdown=False):  # topdown=False makes the walk visit subdirectories first
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)  # remove the root directory itself


def add_constraint(atoms, constraint):
    consts = atoms.constraints
    if len(consts) == 0:
        atoms.set_constraint(constraint)
    else:
        consts.append(constraint)
        atoms.set_constraint(consts)

def add_bond_constraints(atoms, indices):
    try:
        assert len(indices) % 2 == 0
    except:
        raise ValueError("Uneven number of indices")
    nPairs = int(len(indices)/2)
    for i in range(nPairs):
        add_constraint(atoms, FixBondLength(indices[2*i], indices[1+(2*i)]))

def write_contcar(atoms, root):
    atoms.write(os.path.join(root, 'CONTCAR'), format="vasp", direct=True)
    insert_el(os.path.join(root, 'CONTCAR'))

def optimizer(atoms, root, opter, opt_alpha=150):
    traj = opj(root, "opt.traj")
    log = opj(root, "opt.log")
    restart = opj(root, "hessian.pckl")
    dyn = opter(atoms, trajectory=traj, logfile=log, restart=restart, a=(opt_alpha / 70) * 0.1)
    return dyn


def get_inputs_list(fname, auto_lower=True):
    inputs = []
    with open(fname, "r") as f:
        for line in f:
            key = line.split(":")[0]
            val = line.rstrip("\n").split(":")[1]
            if "#" in val:
                val = val[:val.index("#")]
            if auto_lower:
                key = key.lower()
                val = val.lower()
            if not "#" in key:
                inputs.append(tuple([key, val]))
    return inputs

def fix_work_dir(work_dir):
    if work_dir is None:
        work_dir = os.getcwd()
    if work_dir[-1] != "/":
        work_dir += "/"
    return work_dir

def get_bond_length(atoms, indices):
    posn1 = atoms.positions[indices[0]]
    posn2 = atoms.positions[indices[1]]
    return np.linalg.norm(posn2 - posn1)

def step_bond_with_momentum(atom_pair, step_length, atoms_prev_2, atoms_prev_1):
    target_length = get_bond_length(atoms_prev_1, atom_pair) + step_length
    dir_vecs = []
    for i in range(len(atoms_prev_1.positions)):
        dir_vecs.append(atoms_prev_1[i] - atoms_prev_2[i])
    for i in range(len(dir_vecs)):
        atoms_prev_1.positions[i] += dir_vecs[i]
    dir_vec = atoms_prev_1.positions[atom_pair[1]] - atoms_prev_1.positions[atom_pair[0]]
    cur_length = np.linalg.norm(dir_vec)
    should_be_0 = target_length - cur_length
    if not np.isclose(should_be_0, 0.0):
        atoms_prev_1.positions[atom_pair[1]] += dir_vec*(should_be_0)/np.linalg.norm(dir_vec)
    return atoms_prev_1


def dump_template_input(fname, template, cwd):
    dump_str = ""
    for el in template:
        dump_str += el + "\n"
    with open(opj(cwd, fname), "w") as f:
        f.write(dump_str)
