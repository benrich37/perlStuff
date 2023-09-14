import os
from os.path import join as opj, exists as ope
from ase.io.trajectory import TrajectoryReader

import numpy as np

from helpers.generic_helpers import get_scan_atoms_list, log_def, \
    get_atoms_list_from_out, get_do_cell, is_done, get_charges


logx_init_str = "\n Entering Link 1 \n \n"
logx_finish_str = " Normal termination of Gaussian 16"


def get_scan_logx_str(scan_dir, e_conv=(1/27.211397)):
    atoms_list = get_scan_atoms_list(scan_dir)
    dump_str = logx_init_str
    for i in range(len(atoms_list)):
        dump_str += log_input_orientation(atoms_list[i], do_cell=True)
        dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
        dump_str += log_charges(atoms_list[i])
        dump_str += opt_spacer(i, len(atoms_list))
    dump_str += logx_finish_str
    return dump_str


def write_scan_logx(scan_dir, log_fn=log_def):
    scan_logx = opj(scan_dir, "scan.logx")
    log_fn(f"Updating scan logx to {scan_logx}")
    with open(scan_logx, "w") as f:
        f.write(get_scan_logx_str(scan_dir))
        f.close()



def log_forces(atoms):
    dump_str = ""
    dump_str += "-------------------------------------------------------------------\n"
    dump_str += " Center     Atomic                   Forces (Hartrees/Bohr)\n"
    dump_str += " Number     Number              X              Y              Z\n"
    dump_str += " -------------------------------------------------------------------\n"
    forces = []
    try:
        momenta = atoms.get_momenta()
    except Exception as e:
        print(e)
        momenta = np.zeros([len(atoms.get_atomic_numbers()), 3])
    for i, number in enumerate(atoms.get_atomic_numbers()):
        add_str = f" {i+1} {number}"
        force = momenta[i]
        forces.append(np.linalg.norm(force))
        for j in range(3):
            add_str += f"\t{force[j]:.9f}"
        add_str += "\n"
        dump_str += add_str
    dump_str += " -------------------------------------------------------------------\n"
    forces = np.array(forces)
    dump_str += f" Cartesian Forces:  Max {max(forces):.9f} RMS {np.std(forces):.9f}\n"
    return dump_str


def out_to_logx_str(outfile, use_force=False, e_conv=(1/27.211397)):
    atoms_list = get_atoms_list_from_out(outfile)
    dump_str = logx_init_str
    do_cell = get_do_cell(atoms_list[0].cell)
    if use_force:
        do_cell = False
    for i in range(len(atoms_list)):
        dump_str += log_input_orientation(atoms_list[i], do_cell=do_cell)
        dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
        dump_str += log_charges(atoms_list[i])
        dump_str += log_forces(atoms_list[i])
        dump_str += opt_spacer(i, len(atoms_list))
    if is_done(outfile):
        dump_str += log_input_orientation(atoms_list[-1])
        dump_str += logx_finish_str
    return dump_str


def traj_to_log_str_helper(traj, start_step=0, maxSteps=1000):
    dump_str = ""
    nSteps = len(traj)
    do_cell = np.sum(abs(traj[0].cell)) > 0
    for i in range(nSteps):
        dump_str += log_input_orientation(traj[i], do_cell=do_cell)
        try:
            dump_str += scf_str(traj[i])
        except:
            pass
        try:
            dump_str += log_charges(traj[i])
        except:
            pass
        dump_str += opt_spacer(start_step + i, maxSteps)
    return dump_str

def traj_to_log_str(traj):
    dump_str = logx_init_str
    dump_str += traj_to_log_str_helper(traj, start_step=0, maxSteps=len(traj))
    dump_str += log_input_orientation(traj[-1])
    dump_str += logx_finish_str
    return dump_str


def get_last_step(logxfname):
    find_str = "number"
    last_step = 0
    with open(logxfname, "r") as f:
        for line in f:
            if "Step number" in line:
                last_step = int(line[line.index(find_str) + len(find_str):].strip().split()[0])
    return last_step



def get_append_traj_logx_str(traj, logxfname):
    start_step = get_last_step(logxfname) + 1
    append_str = traj_to_log_str_helper(traj, start_step=start_step, maxSteps=1000)
    return append_str

def terminate_logx(logxfname):
    with open(logxfname, "a") as f:
        f.write(logx_finish_str)

def traj_to_logx_appendable(trajfname, logxfname):
    traj = TrajectoryReader(trajfname)
    if not ope(logxfname):
        with open(logxfname, "w") as f:
            f.write(logx_init_str)
    append_str = get_append_traj_logx_str(traj, logxfname)
    with open(logxfname, "a") as f:
        f.write(append_str)


def log_charges(atoms):
    try:
        charges = get_charges(atoms)
        nAtoms = len(atoms.positions)
        symbols = atoms.get_chemical_symbols()
    except:
        return " "
    dump_str = " **********************************************************************\n\n"
    dump_str += "            Population analysis using the SCF Density.\n\n"
    dump_str = " **********************************************************************\n\n Mulliken charges:\n    1\n"
    for i in range(nAtoms):
        dump_str += f"{int(i+1)} {symbols[i]} {charges[i]} \n"
    dump_str += f" Sum of Mulliken charges = {np.sum(charges)}\n"
    return dump_str


def log_input_orientation(atoms, do_cell=False):
    dump_str = "                          Input orientation:                          \n"
    dump_str += " ---------------------------------------------------------------------\n"
    dump_str += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    dump_str += " Number     Number       Type             X           Y           Z\n"
    dump_str += " ---------------------------------------------------------------------\n"
    at_ns = atoms.get_atomic_numbers()
    at_posns = atoms.positions
    nAtoms = len(at_ns)
    for i in range(nAtoms):
        dump_str += f" {i+1} {at_ns[i]} 0 "
        for j in range(3):
            dump_str += f"{at_posns[i][j]} "
        dump_str += "\n"
    if do_cell:
        cell = atoms.cell
        for i in range(3):
            dump_str += f"{i + nAtoms + 1} -2 0 "
            for j in range(3):
                dump_str += f"{cell[i][j]} "
            dump_str += "\n"
    dump_str += " ---------------------------------------------------------------------\n"
    return dump_str


def opt_spacer(i, nSteps):
    dump_str = "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    dump_str += f"\n Step number   {i}\n"
    if i == nSteps:
        dump_str += " Optimization completed.\n"
        dump_str += "    -- Stationary point found.\n"
    dump_str += "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    return dump_str


def scf_str(atoms, e_conv=(1/27.211397)):
    try:
        E = atoms.E
    except:
        E = atoms.get_potential_energy()
    return f"\n SCF Done:  E =  {E*e_conv}\n\n"


def out_to_logx(save_dir, outfile, log_fn=lambda s: print(s)):
    try:
        fname = opj(save_dir, "out.logx")
        with open(fname, "w") as f:
            f.write(out_to_logx_str(outfile))
        f.close()
    except Exception as e:
        log_fn(e)
        pass
    try:
        fname = opj(save_dir, "out_wforce.logx")
        with open(fname, "w") as f:
            f.write(out_to_logx_str(outfile, use_force=True))
        f.close()
    except Exception as e:
        log_fn(e)
        pass


def _write_logx(atoms, fname, do_cell=True, do_charges=True):
    fb = os.path.basename(fname)
    if ("." in fb):
        force_fname = opj(os.path.dirname(fname), f"{fb.split('.')[-2]}_wforce.{fb.split('.')[-1]}")
    else:
        force_fname = opj(os.path.dirname(fname), f"{fb}_wforce.logx")
    #######
    if not ope(fname):
        with open(fname, "w") as f:
            f.write(logx_init_str)
    nLast = get_last_step(fname)
    step = nLast + 1
    with open(fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=do_cell))
        f.write(scf_str(atoms))
        if do_charges:
            f.write(log_charges(atoms))
        f.write(opt_spacer(step, 100000))
    ########
    if not ope(force_fname):
        with open(force_fname, "w") as f:
            f.write(logx_init_str)
    nLast = get_last_step(force_fname)
    step = nLast + 1
    with open(force_fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=False))
        f.write(scf_str(atoms))
        if do_charges:
            f.write(log_charges(atoms))
        f.write(log_forces(atoms))
        f.write(opt_spacer(step, 100000))


def finished_logx(atoms, fname, step, maxstep, do_cell=True):
    with open(fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=do_cell))
        f.write(scf_str(atoms))
        f.write(log_charges(atoms))
        f.write(opt_spacer(step, maxstep))
        f.write(logx_finish_str)


def sp_logx(atoms, fname, do_cell=True):
    if ope(fname):
        os.remove(fname)
    dump_str = logx_init_str
    dump_str += log_input_orientation(atoms, do_cell=do_cell)
    dump_str += scf_str(atoms)
    dump_str += log_charges(atoms)
    dump_str += logx_finish_str
    with open(fname, "w") as f:
        f.write(dump_str)

def get_opt_dot_log_faker_str(atoms_list):
    dump_str = "\n Running Convergence Step: 1 \n"
    dump_str += "      Step     Time          Energy         fmax \n"
    dump_str += "*Force-consistent energies used in optimization.\n"
    for i, atoms in enumerate(atoms_list):
        forces = []
        for force in atoms.get_momenta():
            forces.append(np.linalg.norm(force))
        dump_str += f"JDFT:   {i} 00:00:00   {atoms.E:.6f}*       {max(forces):.4f}\n"
    dump_str += "\n"
    return dump_str

def opt_dot_log_faker(outfile, save_dir):
    atoms_list = get_atoms_list_from_out(outfile)
    dump_str = get_opt_dot_log_faker_str(atoms_list)
    optdotlog = opj(save_dir, "opt.log")
    with open(optdotlog, "w") as f:
        f.write(dump_str)
        f.close()

