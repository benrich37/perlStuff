import sys
from ase.io.trajectory import TrajectoryReader
import numpy as np
import argparse

def traj_to_log_str(traj):
    dump_str = "\n Entering Link 1 \n \n"
    nSteps = len(traj)
    do_cell = np.sum(abs(traj[0].cell)) > 0
    for i in range(nSteps):
        dump_str += log_input_orientation(traj[i], do_cell=do_cell)
        dump_str += scf_str(traj[i])
        dump_str += opt_spacer(i, nSteps)
        dump_str += log_charges(traj[i])
    dump_str += log_input_orientation(traj[-1])
    dump_str += " Normal termination of Gaussian 16"
    return dump_str

def scf_str(atoms, e_conv=(1/27.211397)):
    return f"\n SCF Done:  E =  {atoms.get_potential_energy()*e_conv}\n\n"

def opt_spacer(i, nSteps):
    dump_str = "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    dump_str += f"\n Step number   {i+1}\n"
    if i == nSteps:
        dump_str += " Optimization completed.\n"
        dump_str += "    -- Stationary point found.\n"
    dump_str += "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    return dump_str

def log_input_orientation(atoms, do_cell=False):
    dump_str = "                          Input orientation:                          \n"
    dump_str += " ---------------------------------------------------------------------\n"
    dump_str += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    dump_str += " Number     Number       Type             X           Y           Z\n"
    dump_str += " ---------------------------------------------------------------------"
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

def log_charges(atoms):
    charges = atoms.charges
    nAtoms = len(atoms.positions)
    symbols = atoms.get_chemical_symbols()
    dump_str = " **********************************************************************\n\n"
    dump_str += "            Population analysis using the SCF Density.\n\n"
    dump_str = " **********************************************************************\n\n Mulliken charges:\n    1\n"
    for i in range(nAtoms):
        dump_str += f"{int(i+1)} {symbols[i]} {charges[i]} \n"
    dump_str += f" Sum of Mulliken charges = {np.sum(charges)}\n"
    return dump_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input traj file")
    args = parser.parse_args()
    file = args.input
    assert ".traj" in file
    traj = TrajectoryReader(file)
    with open(file[:file.index(".traj")] + "_traj.logx", "w") as f:
        f.write(traj_to_log_str(traj))
        f.close()

