import os
from os.path import join as opj, exists as ope

import numpy as np

from helpers.generic_helpers import get_scan_atoms_list, log_def, \
    get_atoms_list_from_out, get_do_cell, is_done, get_charges


def get_scan_logx_str(scan_dir, e_conv=(1/27.211397)):
    atoms_list = get_scan_atoms_list(scan_dir)
    dump_str = "\n Entering Link 1 \n \n"
    for i in range(len(atoms_list)):
        dump_str += log_input_orientation(atoms_list[i], do_cell=True)
        dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
        dump_str += log_charges(atoms_list[i])
        dump_str += opt_spacer(i, len(atoms_list))
    dump_str += " Normal termination of Gaussian 16"
    return dump_str


def write_scan_logx(scan_dir, log_fn=log_def):
    scan_logx = opj(scan_dir, "scan.logx")
    log_fn(f"Updating scan logx to {scan_logx}")
    with open(scan_logx, "w") as f:
        f.write(get_scan_logx_str(scan_dir))
        f.close()


def out_to_logx_str(outfile, e_conv=(1/27.211397)):
    atoms_list = get_atoms_list_from_out(outfile)
    dump_str = "\n Entering Link 1 \n \n"
    do_cell = get_do_cell(atoms_list[0].cell)
    for i in range(len(atoms_list)):
        dump_str += log_input_orientation(atoms_list[i], do_cell=do_cell)
        dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
        dump_str += log_charges(atoms_list[i])
        dump_str += opt_spacer(i, len(atoms_list))
    if is_done(outfile):
        dump_str += log_input_orientation(atoms_list[-1])
        dump_str += " Normal termination of Gaussian 16"
    return dump_str


def traj_to_log_str(traj):
    dump_str = "\n Entering Link 1 \n \n"
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
        dump_str += opt_spacer(i, nSteps)
    dump_str += log_input_orientation(traj[-1])
    dump_str += " Normal termination of Gaussian 16"
    return dump_str


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
    dump_str += f"\n Step number   {i+1}\n"
    if i == nSteps:
        dump_str += " Optimization completed.\n"
        dump_str += "    -- Stationary point found.\n"
    dump_str += "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    return dump_str


def scf_str(atoms, e_conv=(1/27.211397)):
    return f"\n SCF Done:  E =  {atoms.get_potential_energy()*e_conv}\n\n"


def out_to_logx(save_dir, outfile, log_fn=lambda s: print(s)):
    try:
        fname = opj(save_dir, "out.logx")
        with open(fname, "w") as f:
            f.write(out_to_logx_str(outfile))
        f.close()
    except Exception as e:
        log_fn(e)
        pass


def _write_logx(atoms, fname, dyn, maxstep, do_cell=True, do_charges=True):
    if not ope(fname):
        with open(fname, "w") as f:
            f.write("\n Entering Link 1 \n \n")
    step = dyn.nsteps
    with open(fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=do_cell))
        f.write(scf_str(atoms))
        if do_charges:
            f.write(log_charges(atoms))
        f.write(opt_spacer(step, maxstep))


def finished_logx(atoms, fname, step, maxstep, do_cell=True):
    with open(fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=do_cell))
        f.write(scf_str(atoms))
        f.write(log_charges(atoms))
        f.write(opt_spacer(step, maxstep))
        f.write("\n Normal termination of Gaussian 16 at Fri Jul 21 12:28:14 2023.\n")


def sp_logx(atoms, fname, do_cell=True):
    if ope(fname):
        os.remove(fname)
    dump_str = "\n Entering Link 1 \n \n"
    dump_str += log_input_orientation(atoms, do_cell=do_cell)
    dump_str += scf_str(atoms)
    dump_str += log_charges(atoms)
    dump_str += "\n Normal termination of Gaussian 16 at Fri Jul 21 12:28:14 2023.\n"
    with open(fname, "w") as f:
        f.write(dump_str)