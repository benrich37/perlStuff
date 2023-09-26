import os
from os.path import exists as ope, join as opj
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from datetime import datetime
from helpers.generic_helpers import get_cmds_list, get_inputs_list, fix_work_dir, optimizer, remove_dir_recursive, \
    get_atoms_list_from_out, get_do_cell
from helpers.generic_helpers import _write_contcar, get_log_fn, dump_template_input, read_pbc_val
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.generic_helpers import check_submit, get_atoms_from_coords_out, add_dos_cmds, add_vib_cmds
from helpers.generic_helpers import copy_best_state_files, has_coords_out_files
from helpers.generic_helpers import _write_opt_iolog, check_for_restart, log_def, check_structure, log_and_abort
from helpers.logx_helpers import out_to_logx, _write_logx, finished_logx, sp_logx
from sys import exit, stderr
from shutil import copy as cp


opt_template = ["structure: CONTCAR # Structure for optimization",
                "gpu: True # Whether or not to use GPU (much faster)",
                "pbc: False False False # Periodic boundary conditions for unit cell",
                "freeze base: True # Whether to freeze lower atoms",
                "freeze tol: 3. # Distance from topmost atom to impose freeze cutoff for freeze base",
                "# save DOS: True # save DOS output from JDFTx",
                "save pDOS: True # Save pDOS output from JDFTx (overrides input for save DOS)"]


def read_opt_inputs(fname = "vib_input"):
    work_dir = None
    structure = None
    if not ope(fname):
        if not ope("opt_input"):
            dump_template_input(fname, opt_template, os.getcwd())
            raise ValueError(f"No opt input supplied: dumping template {fname}")
        else:
            fname = "opt_input"
    inputs = get_inputs_list(fname, auto_lower=False)
    gpu = True
    pbc = [True, True, False]
    freeze_base = False
    freeze_tol = 0.
    save_dos = True
    save_pdos = True
    for input in inputs:
        key, val = input[0], input[1]
        if "structure" in key:
            structure = val.strip()
        if "work" in key:
            work_dir = val
        if "gpu" in key:
            gpu = "true" in val.lower()
        if "pbc" in key:
            pbc = read_pbc_val(val)
        if ("freeze" in key):
            if ("base" in key):
                freeze_base = "true" in val.lower()
            elif ("tol" in key):
                freeze_tol = float(val)
        if ("save" in key):
            if ("dos" in key.lower()):
                save_dos = "true" in val.lower()
            if ("pdos" in key.lower()):
                save_pdos = "true" in val.lower()
    work_dir = fix_work_dir(work_dir)
    return work_dir, structure, gpu, pbc, freeze_base, freeze_tol, save_dos, save_pdos


def finished(dirname):
    with open(opj(dirname, "finished.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")


def get_atoms_from_lat_dir(dir):
    ionpos = opj(dir, "ionpos")
    lattice = opj(dir, "lattice")
    return get_atoms_from_coords_out(ionpos, lattice)


def get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
    if ope(opt_dir):
        if not use_jdft:
            if ope(opj(opt_dir, "CONTCAR")):
                structure = opj(opt_dir, "CONTCAR")
                log_fn(f"Found {structure} for restart structure")
        else:
            outfile = opj(opt_dir, "out")
            if ope(outfile):
                structure = opj(opt_dir, "POSCAR")
                atoms_obj = get_atoms_list_from_out(outfile)[-1]
                write(structure, atoms_obj, format="vasp")
    elif ope(lat_dir):
        os.mkdir(opt_dir)
        if not has_coords_out_files(lat_dir):
            log_fn(f"No ionpos and/or lattice found in {lat_dir}")
            lat_out = opj(lat_dir, "out")
            if ope(lat_out):
                log_fn(f"Reading recent structure from out file in {lat_out}")
                atoms = get_atoms_list_from_out(lat_out)[-1]
                structure = opj(lat_dir, "POSCAR_lat_out")
                log_fn(f"Saving read structure to {structure}")
                write(structure, atoms, format="vasp")
        else:
            log_fn(f"Reading structure from {lat_dir}")
            atoms = get_atoms_from_lat_dir(lat_dir)
            structure = opj(lat_dir, "POSCAR_coords_out")
            log_fn(f"Saving read structure to {structure}")
            write(structure, atoms, format="vasp")
    else:
        os.mkdir(lat_dir)
        os.mkdir(opt_dir)
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    return structure, restart


def get_structure(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=log_def):
    dirs_list = [opt_dir]
    if lat_iters > 0:
        log_fn(f"Lattice opt requested ({lat_iters} iterations) - adding lat dir to setup list")
        dirs_list.append(lat_dir)
    if not restart:
        for d in dirs_list:
            if ope(d):
                log_fn(f"Resetting {d}")
                remove_dir_recursive(d)
            os.mkdir(d)
        if ope(structure):
            log_fn(f"Found {structure} for structure")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    else:
        structure, restart = get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_fn)
    return structure, restart


def freeze_surf_base(atoms, ztol = 3.):
    min_z = min(atoms.positions[:, 2])
    mask = (atoms.positions[:, 2] < (min_z + ztol))
    c = FixAtoms(mask = mask)
    atoms.set_constraint(c)
    return atoms


def run_lat_opt_runner(atoms, structure, lat_dir, root, calc_fn, freeze_base = False, freeze_tol = 0., log_fn=log_def):
    if freeze_base:
        atoms = freeze_surf_base(atoms, ztol=freeze_tol)
    atoms.set_calculator(calc_fn(lat_dir))
    log_fn("lattice optimization starting")
    atoms.get_forces()
    ionpos = opj(lat_dir, "ionpos")
    lattice = opj(lat_dir, "lattice")
    pbc = atoms.pbc
    atoms = get_atoms_from_coords_out(ionpos, lattice)
    atoms.pbc = pbc
    structure = opj(root, structure + "_lat_opted")
    write(structure, atoms, format="vasp")
    log_fn(f"Finished lattice optimization")
    finished(lat_dir)
    return atoms, structure


def run_lat_opt(atoms, structure, lat_dir, root, calc_fn, freeze_base = False, freeze_tol = 0., log_fn=log_def, _failed_before=False):
    run_again = False
    try:
        atoms, structure = run_lat_opt_runner(atoms, structure, lat_dir, root, calc_fn, freeze_base = freeze_base, freeze_tol = freeze_tol, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, lat_dir, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms, structure = run_lat_opt(atoms, structure, lat_dir, root, calc_fn, freeze_base = freeze_base, freeze_tol = freeze_tol, log_fn=log_fn, _failed_before=True)
    return atoms, structure

def get_vib_data(outfile):
    collecting = False
    look_key = "Vibrational free energy components"
    splitter = ":"
    data = {}
    with open(outfile, "r") as f:
        for line in f:
            if collecting:
                if splitter in line:
                    key = line.split(splitter)[0].strip()
                    val = line.split(splitter)[1].strip()
                    data[key] = val
                else:
                    collecting = False
            if look_key in line:
                collecting = True
    return data

import json

def run_sp_runner(atoms_obj, vib_dir_path, calc_fn, freeze_base=False, freeze_tol=0., log_fn=log_def):
    if freeze_base:
        atoms_obj = freeze_surf_base(atoms_obj, ztol=freeze_tol)
    atoms_obj.set_calculator(calc_fn(vib_dir_path))
    log_fn("Single point calculation starting")
    atoms_obj.get_forces()
    outfile = opj(vib_dir_path, "out")
    data = get_vib_data(outfile)
    vibjson = opj(vib_dir_path, "vib_data.json")
    with open(vibjson, "w") as f:
        json.dump(data, f)
    f.close()
    return atoms_obj


def run_sp(atoms_obj, vib_dir_path, root_path, calc_fn, freeze_base = False, freeze_tol = 0., _failed_before=False, log_fn=log_def):
    run_again = False
    try:
        atoms_obj = run_sp_runner(atoms_obj, vib_dir_path, calc_fn, freeze_base=freeze_base, freeze_tol=freeze_tol,
                                  log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, vib_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms_obj = run_sp(atoms_obj, vib_dir_path, root_path, calc_fn, _failed_before=False, log_fn=log_fn)
    return atoms_obj


def run_ase_opt_runner(atoms, root, opter, fmax, max_steps, freeze_base = False, freeze_tol = 0.,log_fn=log_def):
    if freeze_base:
        atoms = freeze_surf_base(atoms, ztol=freeze_tol)
    do_cell = get_do_cell(atoms.pbc)
    dyn = optimizer(atoms, root, opter)
    traj = Trajectory(opj(root, "opt.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
    logx = opj(root, "opt.logx")
    write_logx = lambda: _write_logx(atoms, logx, do_cell=do_cell)
    write_contcar = lambda: _write_contcar(atoms, root)
    write_opt_log = lambda: _write_opt_iolog(atoms, dyn, max_steps, log_fn)
    dyn.attach(traj.write, interval=1)
    dyn.attach(write_contcar, interval=1)
    dyn.attach(write_logx, interval=1)
    dyn.attach(write_opt_log, interval=1)
    log_fn("Optimization starting")
    log_fn(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
    finished_logx(atoms, logx, dyn.nsteps, max_steps)
    sp_logx(atoms, "sp.logx", do_cell=do_cell)
    finished(root)


def run_ase_opt(atoms, opt_dir, opter, calc_fn, fmax, max_steps, freeze_base = False, freeze_tol = 0.,log_fn=log_def, _failed_before=False):
    atoms.set_calculator(calc_fn(opt_dir))
    run_again = False
    try:
        run_ase_opt_runner(atoms, opt_dir, opter, fmax, max_steps, freeze_base = freeze_base, freeze_tol = freeze_tol, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, opt_dir, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        run_ase_opt(atoms, opt_dir, opter, calc_fn, fmax, max_steps, freeze_base = freeze_base, freeze_tol = freeze_tol, log_fn=log_fn, _failed_before=True)

def copy_result_files(opt_dir, work_dir):
    result_files = ["CONTCAR", "CONTCAR.gjf", "Ecomponents", "out"]
    for file in result_files:
        full = opj(opt_dir, file)
        if ope(full):
            cp(full, work_dir)


def append_out_to_logx(outfile, logx, log_fn=log_def):
    log_fn(f"Gathering minimization structures from existing out file")
    atoms_list = get_atoms_list_from_out(outfile)
    log_fn(f"{len(atoms_list)} found from out file")
    do_cell = get_do_cell(atoms_list[-1].pbc)
    for atoms in atoms_list:
        _write_logx(atoms, logx, do_cell=do_cell)

def make_jdft_logx(opt_dir, log_fn=log_def):
    outfile = opj(opt_dir, "out")
    logx = opj(opt_dir, "out.logx")
    if ope(logx):
        log_fn("Appending existing out file data to existing logx file")
        append_out_to_logx(outfile, logx, log_fn=log_fn)
    elif ope(outfile):
        out_to_logx(opt_dir, outfile, log_fn=log_fn)
        log_fn("Writing existing out file to logx file")



from os import mkdir



def main():
    work_dir, structure, gpu, pbc, freeze_base, freeze_tol, save_dos, save_pdos = read_opt_inputs()
    os.chdir(work_dir)
    opt_log = get_log_fn(work_dir, "vib", False, restart=True)
    opt_log("Scanning/setting up dirs")
    vib_dir = opj(work_dir, "vib")
    if not ope(vib_dir):
        mkdir(vib_dir)
    opt_dir = opj(work_dir, "ion_opt")
    lat_dir = opj(work_dir, "lat_opt")
    opt_log("Getting structure path")
    structure = opj(work_dir, structure)
    structure = check_structure(structure, work_dir, log_fn=opt_log)
    # structure, restart = get_structure(structure, True, work_dir, opt_dir, lat_dir, 0, True)
    opt_log("Organizing JDFTx commands")
    exe_cmd = get_exe_cmd(gpu, opt_log)
    cmds = get_cmds_list(work_dir, ref_struct=structure)
    opt_log("Reading structure")
    atoms = read(structure, format="vasp")
    opt_log("Adding single point special commands for calculation")
    cmds = add_dos_cmds(cmds, atoms, save_dos, save_pdos)
    cmds = add_vib_cmds(cmds)
    opt_log(f"Setting {structure} to atoms object")
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, log_fn=opt_log)
    opt_log(f"Finding/copying any state files to {vib_dir}")
    copy_best_state_files([work_dir, lat_dir, opt_dir], vib_dir, log_fn=opt_log)
    check_submit(gpu, os.getcwd(), "vib", log_fn=opt_log)
    run_sp(atoms, vib_dir, work_dir, get_calc, freeze_base=freeze_base, freeze_tol=freeze_tol, log_fn=opt_log)
    # copy_result_files(opt_dir, work_dir)

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

