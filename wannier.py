import os
from os.path import exists as ope, join as opj, basename
from os import mkdir, listdir
from ase.io import read, write
from datetime import datetime
from helpers.generic_helpers import get_inputs_list, fix_work_dir, remove_dir_recursive, get_atoms_list_from_out, get_cmds_dict
from helpers.generic_helpers import get_log_fn, dump_template_input, read_pbc_val
from helpers.calc_helpers import _get_calc, get_exe_cmd, _get_wannier_calc, get_wannier_exe_cmd
from helpers.generic_helpers import check_submit, add_cohp_cmds, get_atoms_from_out, add_sp_cmds, append_key_val_to_cmds_list
from helpers.generic_helpers import get_ionic_opt_cmds_list, check_for_restart, add_wannier_centers
from helpers.generic_helpers import log_def, check_structure, log_and_abort, cmds_dict_to_list, get_int_dirs
from sys import exit, stderr
from shutil import copy as cp, move as mv
import numpy as np


opt_template = ["structure: POSCAR # Structure for optimization",
                "gpu: True # Whether or not to use GPU (much faster)",
                "pbc: False False False # Periodic boundary conditions for unit cell",
                "pseudoset: GBRV # directory name containing pseudopotentials you wish to use (top directory must be assigned to 'JDFTx_pseudo' environmental variable)",
                "bias: 0.00V # Bias relative to SHE (is only used if 'target-mu *' in inputs file",
                "center: atom 0 px # specify centers for 'wannier-center'",
                "center-pinned: Gaussian 0.5 0.5 0.5 # specify centers for 'wannier-center-pinned'"
                "wannier: localizationMeasure RealSpace # specify key/args for the 'wannier' command"]


job_type_name = "wannier"
def read_opt_inputs(fname = f"{job_type_name}_input"):
    work_dir = None
    structure = None
    if not ope(fname):
        dump_template_input(fname, opt_template, os.getcwd())
        raise ValueError(f"No {job_type_name} input supplied: dumping template {fname}")
    inputs = get_inputs_list(fname, auto_lower=False)
    gpu = True
    pbc = [True, True, False]
    save_state = False
    ortho = True
    pseudoset = "GBRV"
    bias = 0.0
    skip_sp = False
    centers = []
    centers_pinned = []
    wannier_cmds = []
    for input in inputs:
        key, val = input[0], input[1]
        if "center" in key:
            if "pin" in key:
                centers_pinned.append(val)
            else:
                centers.append(val)
        elif "wannier" in key:
            wannier_cmds.append(val)
        if "pseudo" in key:
            pseudoset = val.strip()
        if "structure" in key:
            structure = val.strip()
        if "work" in key:
            work_dir = val
        if "gpu" in key:
            gpu = "true" in val.lower()
        if "pbc" in key:
            pbc = read_pbc_val(val)
        if ("ortho" in key):
            ortho = "true" in val.lower()
        if ("save" in key) and ("state" in key):
            save_state = "true" in val.lower()
        if "bias" in key:
            bias = float(val.strip().rstrip("V"))
        if ("skip" in key) and (("sp" in key) or ("single" in key)):
            skip_sp = "true" in val.lower()
    work_dir = fix_work_dir(work_dir)
    return work_dir, structure, gpu, pbc, ortho, save_state, pseudoset, bias, skip_sp, centers, centers_pinned, wannier_cmds


def finished(dir_path):
    with open(opj(dir_path, f"finished_{basename(dir_path)}.txt"), "w") as f:
        f.write("done")


def is_done(dir_path):
    idx = int(basename(dir_path))
    return ope(opj(dir_path, f"finished_{idx}.txt"))

def run_sp_runner(atoms_obj, sp_dir_path, calc_fn, log_fn=log_def):
    atoms_obj.set_calculator(calc_fn(sp_dir_path))
    log_fn("Single point calculation starting")
    atoms_obj.get_forces()
    outfile = opj(sp_dir_path, "out")
    if ope(outfile):
        finished(sp_dir_path)
    else:
        log_and_abort(f"No output data given - check error file", log_fn=log_fn)
    return atoms_obj

def run_sp(atoms_obj, ion_dir_path, root_path, calc_fn, _failed_before=False, log_fn=log_def):
    run_again = False
    try:
        atoms_obj = run_sp_runner(atoms_obj, ion_dir_path, calc_fn, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, ion_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms_obj = run_sp(atoms_obj, ion_dir_path, root_path, calc_fn, _failed_before=True, log_fn=log_fn)
    return atoms_obj


def run_wannier_runner(atoms_obj, wannier_dir_path, calc_fn, log_fn=log_def):
    fn = calc_fn(wannier_dir_path)
    atoms_obj.set_calculator(fn)
    log_fn("Wannier localization starting")
    try:
        atoms_obj.get_potential_energy()
    except Exception as e:
        print("problem running vannier (l118)")
        print(e)
    outfile = opj(wannier_dir_path, "out")
    if ope(outfile):
        finished(wannier_dir_path)
    else:
        log_and_abort(f"No output data given - check error file", log_fn=log_fn)
    return atoms_obj

def run_wannier(atoms_obj, wannier_dir_path, root_path, calc_fn, _failed_before=False, log_fn=log_def):
    run_again = False
    try:
        atoms_obj = run_wannier_runner(atoms_obj, wannier_dir_path, calc_fn, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before, wannier_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        atoms_obj = run_wannier(atoms_obj, wannier_dir_path, root_path, calc_fn, _failed_before=True, log_fn=log_fn)
    return atoms_obj

def store_wannier(wannier_dir_path, centers, centers_pinned, wan_special_cmds):
    int_dirs = get_int_dirs(wannier_dir_path)
    if len(int_dirs):
        last = int(basename(int_dirs[-1]))
    else:
        last = -1
    cur = last + 1
    store_dir = opj(wannier_dir_path, str(cur))
    mkdir(store_dir)
    fs = listdir(wannier_dir_path)
    for f in fs:
        if "mlwf" in f:
            mv(opj(wannier_dir_path, f), opj(store_dir, f))
    with open(opj(store_dir, "wan_input.txt"), "w") as f:
        out_str = str(centers) + "\n"
        out_str += str(centers_pinned) + "\n"
        out_str += str(wan_special_cmds) + "\n"
        f.write(out_str)
    mv(opj(wannier_dir_path, "out"), opj(store_dir, "out"))
    cp(opj(wannier_dir_path, "in"), opj(store_dir, "in"))


def get_el_idx(atoms, aidx):
    el_idx_list = get_el_idx_list(atoms)
    return el_idx_list[aidx]


def get_el_idx_list(atoms):
    syms = atoms.get_chemical_symbols()
    el_idx_list = []
    count_dict = {}
    for sym in syms:
        if not sym in count_dict:
            count_dict[sym] = 0
        count_dict[sym] += 1
        el_idx_list.append(count_dict[sym])
    return el_idx_list

def get_wan_center_cmd(csplit, atoms):
    aidx = int(csplit[1])
    sym = atoms.get_chemical_symbols()[aidx]
    idx = get_el_idx(atoms, aidx)
    wan_cmd = f"{sym} {idx} " + " ".join(csplit[2:])
    return wan_cmd


def parse_centers(centers, atoms):
    parsed_centers = []
    for c in centers:
        csplit = c.strip().split(" ")
        c_type = csplit[0]
        if "atom" in c_type.lower():
            parsed_centers.append(get_wan_center_cmd(csplit, atoms))
        else:
            parsed_centers.append(c)
    return parsed_centers





def main():
    work_dir, structure, gpu, pbc, ortho, save_state, pseudoSet, bias, skip_sp, centers, centers_pinned, wan_special_cmds = read_opt_inputs()
    os.chdir(work_dir)
    wannier_dir = opj(work_dir, job_type_name)
    if not ope(wannier_dir):
        mkdir(wannier_dir)
    structure = opj(work_dir, structure)
    wannier_log = get_log_fn(work_dir, job_type_name, False, restart=False)
    structure = check_structure(structure, work_dir, log_fn=wannier_log)
    atoms = read(structure, format="vasp")
    centers = parse_centers(centers, atoms)
    centers_pinned = parse_centers(centers_pinned, atoms)
    if not skip_sp:
        sp_exe_cmd = get_exe_cmd(gpu, wannier_log)
        sp_cmds = get_cmds_dict(work_dir, ref_struct=structure, bias=bias, pbc=pbc, log_fn=wannier_log)
        sp_cmds = cmds_dict_to_list(sp_cmds)
        sp_cmds = add_sp_cmds(sp_cmds)
        sp_cmds = add_cohp_cmds(sp_cmds, ortho=ortho)
        sp_ion_cmds = get_ionic_opt_cmds_list(sp_cmds, 0)
        wannier_log(f"Setting {structure} to atoms object")
        get_sp_ion_calc = lambda root: _get_calc(sp_exe_cmd, sp_ion_cmds, root, pseudoSet=pseudoSet, log_fn=wannier_log)
        check_submit(gpu, os.getcwd(), job_type_name, log_fn=wannier_log)
        wannier_log(f"Running single point calculation")
        run_sp(atoms, wannier_dir, work_dir, get_sp_ion_calc, log_fn=wannier_log)
    wannier_exe_cmd = get_wannier_exe_cmd(gpu, log_fn=wannier_log)
    wannier_cmds = get_cmds_dict(work_dir, ref_struct=structure, bias=bias, pbc=pbc, log_fn=wannier_log)
    wannier_cmds = cmds_dict_to_list(wannier_cmds)
    wannier_cmds = add_wannier_centers(wannier_cmds, centers, debug_dens=True)
    wannier_cmds = add_wannier_centers(wannier_cmds, centers_pinned, pin=True)
    if len(wan_special_cmds):
        wannier_cmds = append_key_val_to_cmds_list(wannier_cmds, "wannier", " ".join(wan_special_cmds), allow_duplicates=False)
    get_wannier_calc = lambda root:_get_wannier_calc(wannier_exe_cmd, wannier_cmds, root, pseudoSet=pseudoSet, log_fn=wannier_log)
    run_wannier(atoms, wannier_dir, work_dir, get_wannier_calc, _failed_before=False, log_fn=wannier_log)
    store_wannier(wannier_dir, centers, centers_pinned, wan_special_cmds)

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

