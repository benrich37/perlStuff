import os
from ase.io import read, write
import numpy as np
from generic_helpers import log_generic, atom_str, get_inputs_list, step_bond_with_momentum

def _scan_log(log_str, work, print_bool=False):
    log_generic(log_str, work, "bond_scan", print_bool)


def _prep_input(step_idx, atom_pair, step_length, follow, log_func, work_dir, step_type):
    if not work_dir == os.getcwd():
        os.chdir(work_dir)
    print_str = f"Prepared structure for step {step_idx} with"
    atoms = read(str(step_idx - 1) + "/CONTCAR", format="vasp")
    if step_idx <= 1:
        follow = False
    if follow:
        print_str += " atom momentum followed"
        atoms_prev_2 = read(str(step_idx - 2) + "/CONTCAR", format="vasp")
        atoms_prev_1 = read(str(step_idx - 1) + "/CONTCAR", format="vasp")
        atoms = step_bond_with_momentum(atom_pair, step_length, atoms_prev_2, atoms_prev_1)
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

