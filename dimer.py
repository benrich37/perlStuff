import os
from JDFTx import JDFTx
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from os.path import join as opj
from os.path import exists as ope
import numpy as np
import shutil
from ase.neb import NEB
import time
from helpers.generic_helpers import get_int_dirs, copy_state_files, atom_str, get_cmds, get_int_dirs_indices, \
    get_atoms_list_from_out, get_do_cell
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, _write_contcar, add_bond_constraints, optimizer
from helpers.generic_helpers import dump_template_input, _get_calc, get_exe_cmd, get_log_fn, copy_file, log_def, has_coords_out_files
from helpers.generic_helpers import _write_logx, _write_opt_log, check_for_restart, finished_logx, sp_logx, bond_str
from helpers.generic_helpers import remove_dir_recursive, get_ionic_opt_cmds, check_submit, copy_best_state_f
from helpers.generic_helpers import get_atoms_from_coords_out, out_to_logx, death_by_nan, reset_atoms_death_by_nan, write_scan_logx
from helpers.se_neb_helpers import get_fs, has_max, check_poscar, neb_optimizer, fix_step_size
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate

dimer_template = [ "bond: 1, 5 # (1st atom index (counting from 1 (1-based indexing)), 2nd atom index, number of steps, step size)",
                   "restart: False # ",
                   "max_steps: 100 # max number of steps for scan opts",
                   "fmax: 0.05 # fmax perameter for both neb and scan opt",
                   "pbc: True, true, false # which lattice vectors to impose periodic boundary conditions on",]

def read_dimer_inputs(fname="dimer_inputs"):
    if not ope(fname):
        dump_template_input(fname, dimer_template, os.getcwd())
        raise ValueError(f"No dimer input supplied: dumping template {fname}")
    lookline = None
    max_steps = 100
    fmax = 0.01
    work_dir = None
    inputs = get_inputs_list(fname)
    pbc = [True, True, False]
    restart = False
    for input in inputs:
        key, val = input[0], input[1]
        if "bond" in key:
            lookline = val.strip().split(",")
        if "work" in key:
            work_dir = val.strip()
        if "max" in key:
            if "steps" in key:
                max_steps = int(val.strip())
            elif ("force" in key) or ("fmax" in key):
                fmax = float(val.strip())
        if "pbc" in key:
            pbc = read_pbc_val(val)
        if "restart" in key:
            restart = "true" in val.lower()
    if not lookline is None:
        atom_pair = [int(lookline[0]) - 1, int(lookline[1]) - 1] # Convert to 0-based indexing
    work_dir = fix_work_dir(work_dir)
    return atom_pair, work_dir, max_steps, fmax, pbc, restart

def get_d_vector(atoms, atom_pair, d_mag=0.1):
    posns = atoms.positions
    p1 = posns[atom_pair[0]]
    p2 = posns[atom_pair[1]]
    v = p2 - p1
    v *= (1/np.linalg.norm(v))*(d_mag/2.)
    d_vector_array = np.zeros(np.shape(posns))
    d_vector_array[atom_pair[0]] = v
    d_vector_array[atom_pair[1]] = -v
    return d_vector_array


if __name__ == '__main__':
    atom_pair, work_dir, max_steps, fmax, pbc, restart = read_dimer_inputs()
    gpu = True  # Make this an input argument eventually
    os.chdir(work_dir)
    dimer_log = get_log_fn(work_dir, "dimer", False, restart=restart)
    dimer_dir = opj(work_dir, "dimer")
    if not ope(dimer_dir):
        dimer_log("Creating dimer directory")
        os.mkdir(dimer_dir)
    check_poscar(work_dir, dimer_log)
    cmds = get_cmds(work_dir, ref_struct="POSCAR")
    exe_cmd = get_exe_cmd(True, dimer_log)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, JDFTx, debug=False, log_fn=dimer_log)
    if restart:
        if ope(opj(dimer_dir, "CONTCAR")):
            dimer_log("Found restart CONTCAR")
            atoms = read(opj(dimer_dir, "CONTCAR"))
        else:
            dimer_log("Could not find restart CONTCAR in dimer dir - setting restart to false and using POSCAR instead")
            restart = False
            atoms = read(opj(work_dir, "POSCAR"), format="vasp")
    else:
        atoms = read(opj(work_dir, "POSCAR"), format="vasp")
    if not restart:
        dimer_log("Copying over available state files")
        copy_best_state_f([work_dir, work_dir], dimer_dir, log_fn=dimer_log)
    else:
        copy_best_state_f([work_dir, dimer_dir], dimer_dir, log_fn=dimer_log)
    dimer_log("attaching calculator")
    atoms.set_calculator(get_calc(dimer_dir))
    atoms.pbc = pbc
    dimer_log("Creating starting displacement vector")
    d_vector = get_d_vector(atoms, atom_pair)
    dimer_log("Creating dimer control object")
    d_control = DimerControl(initial_eigenmode_method='displacement', displacement_method='vector',
                             logfile=opj(dimer_dir, "dimercontrol.log"))
    dimer_log("Creating minmode atoms object")
    d_atoms = MinModeAtoms(atoms, d_control)
    dimer_log("Applying initial displacement")
    d_atoms.displace(displacement_vector=d_vector)
    traj = Trajectory(opj(dimer_dir, "dimer.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
    dimer_log("Creating minmode translate object")
    dim_rlx = MinModeTranslate(d_atoms, trajectory=opj(dimer_dir,"minmodetranslate.traj"), logfile=opj(dimer_dir, "dimercontrol.log"))
    dimer_log("Attaching trajectory logger to minmode translate object")
    dim_rlx.attach(traj.write, interval=1)
    dimer_log("Attaching CONTCAR logger to minmode translate object")
    dim_rlx.attach(lambda: _write_contcar(atoms, dimer_dir), interval=1)
    check_submit(gpu, os.getcwd(), "dimer", log_fn=dimer_log)
    dimer_log("Running dimer")
    dim_rlx.run(fmax=fmax)