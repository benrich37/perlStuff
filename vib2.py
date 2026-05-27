import os
from os.path import exists as ope, join as opj
from ase.io import read, write as _write
from datetime import datetime
from helpers.generic_helpers import get_inputs_list, fix_work_dir, remove_dir_recursive, \
    get_atoms_list_from_out, get_do_cell, get_cmds_dict, get_apply_freeze_func
from helpers.generic_helpers import get_log_fn, dump_template_input, read_pbc_val
from helpers.calc_helpers import get_exe_cmd, _get_calc_new
from helpers.generic_helpers import add_cohp_cmds, get_atoms_from_out, add_elec_density_dump
from helpers.generic_helpers import log_def, check_structure, log_and_abort, cmds_dict_to_list, cmds_list_to_infile
from helpers.logx_helpers import out_to_logx, _write_logx
from scripts.run_ddec6_v3 import main as run_ddec6
from sys import exit, stderr
from os import getcwd
import subprocess
from pymatgen.io.jdftx.inputs import JDFTXInfile
from ase import Atoms
from pathlib import Path

cwd = getcwd()
debug = "perlStuff" in cwd

def write(fname, _atoms, format="vasp"):
    atoms = _atoms.copy()
    atoms.pbc = [True,True,True]
    _write(fname, atoms, format=format)



opt_template = ["structure: POSCAR # Structure for optimization",
                "fmax: 0.04 # Max force for convergence criteria",
                "max_steps: 100 # Max number of steps before exit",
                "gpu: True # Whether or not to use GPU (much faster)",
                "restart: True # Whether to get structure from lat/opt dirs or from input structure",
                "pbc: True True False # Periodic boundary conditions for unit cell",
                "lattice steps: 0 # Number of steps for lattice optimization (0 = no lattice optimization)",
                "opt program: jdft # Which program to use for ionic optimization, options are 'jdft' and 'ase'",
                "freeze base: False # Whether to freeze lower atoms (don't use for bulk calcs)",
                "freeze tol: 3. # Distance from topmost atom to impose freeze cutoff for freeze base",
                "freeze count: 0 # freeze the lowest n atoms - overrides freeze tol if greater than 0"
                "pseudoset: GBRV # directory name containing pseudopotentials you wish to use (top directory must be assigned to 'JDFTx_pseudo' environmental variable)",
                "bias: 0.00V # Bias relative to SHE (is only used if 'target-mu *' in inputs file",
                "ddec6: True # Dump Elec Density and perform DDEC6 analysis on it"]



def read_opt_inputs(fname = "opt_input"):
    if not ope(fname):
        dump_template_input(fname, opt_template, os.getcwd())
        raise ValueError(f"No opt input supplied: dumping template {fname}")
    inputs = get_inputs_list(fname, auto_lower=False)
    opt_inputs_dict = {
        "work_dir": None,
        "structure": None,
        "fmax": 0.01,
        "max_steps": 100,
        "gpu": True,
        "restart": False,
        "pbc": [True, True, False],
        "lat_iters": 0,
        "use_jdft": True,
        "freeze_base": False,
        "freeze_tol": 3.,
        "ortho": True,
        "save_state": False,
        "pseudoSet": "GBRV",
        "bias": 0.0,
        "ddec6": True,
        "freeze_count": 0,
        "exclude_freeze_count": 0,
        "direct_coords": False,
        "freeze_map": None,
        "freeze_all_but_map": None,
        "use_cart": False,
    }
    for input in inputs:
        key, val = input[0], input[1]
        if "pseudo" in key:
            opt_inputs_dict["pseudoSet"] = val.strip()
        if "structure" in key:
            opt_inputs_dict["structure"] = val.strip()
        if "work" in key:
            opt_inputs_dict["work_dir"] = val
        if "gpu" in key:
            opt_inputs_dict["gpu"] = "true" in val.lower()
        if "restart" in key:
            opt_inputs_dict["restart"] = "true" in val.lower()
        if "max" in key:
            if "fmax" in key:
                opt_inputs_dict["fmax"] = float(val)
            elif "step" in key:
                opt_inputs_dict["max_steps"] = int(val)
        if "pbc" in key:
            opt_inputs_dict["pbc"] = read_pbc_val(val)
        if "lat" in key:
            try:
                opt_inputs_dict["n_iters"] = int(val)
                opt_inputs_dict["lat_iters"] = opt_inputs_dict["n_iters"]
            except:
                pass
        if ("direct" in key) and ("coord" in key):
            opt_inputs_dict["direct_coords"] = "true" in val.lower()
        if ("opt" in key) and ("progr" in key):
            if "jdft" in val:
                opt_inputs_dict["use_jdft"] = True
            if "ase" in val:
                opt_inputs_dict["use_jdft"] = False
            else:
                pass
        if "cart" in key:
            opt_inputs_dict["use_cart"] = "true" in val.lower()
        if ("freeze" in key):
            if ("base" in key):
                opt_inputs_dict["freeze_base"] = "true" in val.lower()
            elif ("tol" in key):
                opt_inputs_dict["freeze_tol"] = float(val)
            elif ("count" in key):
                if "exclude" in key:
                    opt_inputs_dict["exclude_freeze_count"] = int(val)
                else:
                    opt_inputs_dict["freeze_count"] = int(val)
            elif ("map" in key):
                freeze_dict = parse_dict_indexing(val)
                if "all_but" in key:
                    opt_inputs_dict["freeze_all_but_map"] = freeze_dict
                else:
                    opt_inputs_dict["freeze_map"] = freeze_dict
        if ("ortho" in key):
            opt_inputs_dict["ortho"] = "true" in val.lower()
        if ("save" in key) and ("state" in key):
            opt_inputs_dict["save_state"] = "true" in val.lower()
        if "bias" in key:
            v= val.strip()
            if "V" in v:
                v = v.rstrip("V")
            if "none" in v.lower() or "no" in v.lower():
                opt_inputs_dict["bias"] = None
            else:
                try:
                    opt_inputs_dict["bias"] = float(v)
                except Exception as e:
                    print(e)
                    print("Assigning no bias")
                    opt_inputs_dict["bias"] = None
        if "ddec6" in key:
            opt_inputs_dict["ddec6"] = "true" in val.lower()
    opt_inputs_dict["work_dir"] = fix_work_dir(opt_inputs_dict["work_dir"])
    return opt_inputs_dict
    # return work_dir, structure, fmax, max_steps, gpu, restart, pbc, lat_iters, use_jdft, freeze_base, freeze_tol, ortho, save_state, pseudoset, bias, ddec6, freeze_count

def parse_dict_indexing(line: str):
    # Expected format: (el1, i1, i2, ...), (el2, i1, i2, ...), ...
    pieces = []
    start_idcs = [i for i, v in enumerate(line) if v == "("]
    end_idcs = [i for i, v in enumerate(line) if v == ")"]
    for start, end in zip(start_idcs, end_idcs):
        pieces.append(line[start + 1:end].strip())
    index_dict = {}
    for piece in pieces:
        el, *indices = [v.strip() for v in piece.split(',')]
        index_dict[el.strip()] = [int(i.strip()) for i in indices]
    return index_dict

def finished(dirname):
    with open(opj(dirname, "finished.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Done")

def is_finished(dirname):
    # return ope(opj(dirname, "finished.txt"))
    return (Path(dirname) / "finished.txt").is_file()

def get_atoms_from_lat_dir(dir):
    outfile = opj(dir, "out")
    return get_atoms_from_out(outfile)

def make_dir(dirname):
    if not ope(dirname):
        os.mkdir(dirname)



def get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_def):
    atoms_obj = None
    outfile_path1 = opj(opt_dir, "out")
    outfile_path2 = opj(opt_dir, opj("jdftx_run", "out"))
    if ope(outfile_path2):
        try:
            atoms_obj = get_atoms_from_out(outfile_path2)
            log_fn(f"Atoms object set from {outfile_path2}")
        except Exception as e:
            log_fn(f"Error reading atoms object from {outfile_path2}")
            pass
    if atoms_obj is None:
        if ope(outfile_path1):
            try:
                atoms_obj = get_atoms_from_out(outfile_path1)
                log_fn(f"Atoms object set from {outfile_path1}")
            except Exception as e:
                log_fn(f"Error reading atoms object from {outfile_path1}")
                pass
    return atoms_obj


def get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
    for path in [opt_dir, lat_dir]:
        if not ope(path):
            make_dir(path)
    # If an atoms is found in the ion_opt dir, then an ionic minimization was run
    # (So even if the lattice opt was run, the ion opt will be the most recent)
    atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn)
    if not atoms is None:
        structure = opj(opt_dir, "POSCAR")
        write(structure, atoms, format="vasp")
        return structure, restart
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(lat_dir, log_fn=log_fn)
    if not atoms is None:
        structure = opj(lat_dir, "POSCAR")
        write(structure, atoms, format="vasp")
        return structure, restart
    if atoms is None:
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    return structure, restart


def get_restart_atoms(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_def):
    for path in [opt_dir, lat_dir]:
        if not ope(path):
            make_dir(path)
    # If an atoms is found in the ion_opt dir, then an ionic minimization was run
    # (So even if the lattice opt was run, the ion opt will be the most recent)
    atoms = get_restart_atoms_from_opt_dir(opt_dir, log_fn=log_fn)
    if atoms is None:
        atoms = get_restart_atoms_from_opt_dir(lat_dir, log_fn=log_fn)
    if atoms is None:
        log_fn(f"Could not gather restart structure from {work_dir}")
        if ope(structure):
            log_fn(f"Using {structure} for structure")
            log_fn(f"Changing restart to False")
            restart = False
            log_fn("setting up lattice and opt dir")
            atoms = read(structure, format="vasp")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    return atoms, restart



def get_structure(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=log_def):
    dirs_list = [opt_dir]
    if lat_iters > 0:
        log_fn(f"Lattice opt requested ({lat_iters} iterations) - adding lat dir to setup list")
        dirs_list.append(lat_dir)
        dirs_list.append(opj(lat_dir, "jdftx_run"))
    if not restart:
        for d in dirs_list:
            if ope(d):
                log_fn(f"Resetting {d}")
                remove_dir_recursive(d)
            make_dir(d)
        if ope(structure):
            log_fn(f"Found {structure} for structure")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    else:
        structure, restart = get_restart_structure(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_fn)
    return structure, restart


def get_atoms(structure, restart, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=log_def):
    dirs_list = [opt_dir]
    if lat_iters > 0:
        log_fn(f"Lattice opt requested ({lat_iters} iterations) - adding lat dir to setup list")
        dirs_list.append(lat_dir)
        dirs_list.append(opj(lat_dir, "jdftx_run"))
    if not restart:
        for d in dirs_list:
            if ope(d):
                log_fn(f"Resetting {d}")
                remove_dir_recursive(d)
            make_dir(d)
        if ope(structure):
            log_fn(f"Found {structure} for structure")
        else:
            log_and_abort(f"Requested structure {structure} not found", log_fn=log_fn)
    else:
        atoms, restart = get_restart_atoms(structure, restart, work_dir, opt_dir, lat_dir, use_jdft, log_fn=log_fn)
    return atoms, restart




def run_ion_opt(
        atoms_obj: Atoms, ion_dir_path, calc_fn, apply_freeze_func,
        log_fn=log_def):
    calculator_object = calc_fn(ion_dir_path)
    atoms_obj.set_calculator(calculator_object)
    atoms_obj = apply_freeze_func(atoms_obj)
    log_fn("ionic optimization starting")
    pbc = atoms_obj.pbc
    atoms_obj.get_properties(['energy'])
    log_fn("ionic optimization finished - organizing output data")
    outfile2 = opj(ion_dir_path, opj("jdftx_run", "out"))
    if ope(outfile2):
        atoms_obj = get_atoms_from_out(outfile2)
    else:
        log_and_abort(f"No output data given - check error file", log_fn=log_fn)
    atoms_obj.pbc = pbc
    structure_path = opj(ion_dir_path, "CONTCAR")
    write(structure_path, atoms_obj, format="vasp")
    structure_path = opj(ion_dir_path, "CONTCAR.gjf")
    write(structure_path, atoms_obj, format="gaussian-in")
    finished(ion_dir_path)
    return atoms_obj


def append_out_to_logx(outfile, logx, log_fn=log_def):
    log_fn(f"Gathering minimization structures from existing out file")
    _atoms_list = get_atoms_list_from_out(outfile)
    atoms_list = []
    for atoms in _atoms_list:
        if not atoms is None:
            atoms_list.append(atoms)
    log_fn(f"{len(atoms_list)} found from out file")
    if len(atoms_list):
        do_cell = get_do_cell(atoms_list[-1].pbc)
        for atoms in atoms_list:
            _write_logx(atoms, logx, do_cell=do_cell)

def make_jdft_logx(opt_dir, log_fn=log_def):
    outfile = opj(opt_dir, "out")
    logx = opj(opt_dir, "out.logx")
    if ope(outfile):
        out_to_logx(opt_dir, outfile, log_fn=log_fn)
        log_fn("Writing existing out file to logx file")



def outfile_protect(work_dir):
    calc_dirs = [opj(work_dir, t) for t in ["ion_opt", "lat_opt"]]
    outfiles = [opj(c, "out") for c in calc_dirs]
    for o in outfiles:
        if ope(o):
            if not debug:
                print("DELETING FLUID-EX-CORR LINE FROM FUNCTIONAL - DELETE ME ONCE THIS BUG IS FIXED")
                subprocess.run(f"sed -i '/fluid-ex-corr/d' {o}", shell=True, check=True)

def fix_restart_bug(work_dir, restart):
    outfile = opj(opj(work_dir, "ion_opt"), "out")
    if restart:
        if ope(outfile):
            if not debug:
                subprocess.run(f"sed -i '/fluid-ex-corr/d' {outfile}", shell=True, check=True)
                subprocess.run(f"sed -i '/lda-PZ/d' {outfile}", shell=True, check=True)


def get_ionic_opt_cmds_infile(infile: JDFTXInfile, ion_iters: int, use_jdft: bool):
    if use_jdft:
        if "ionic-minimize" in infile:
            infile["ionic-minimize"]["nIterations"] = int(max(ion_iters, infile["ionic-minimize"].get("nIterations", 0)))
        else:
            infile["ionic-minimize"] = {"nIterations": ion_iters}
    else:
        if not "ionic-minimize" in infile:
            infile["ionic-minimize"] = {"nIterations": 0}
    return infile

def main(debug=False):
    oid = read_opt_inputs()
    work_dir = oid["work_dir"]
    structure = oid["structure"]
    restart = oid["restart"]
    lat_iters = oid["lat_iters"]
    use_jdft = oid["use_jdft"]
    gpu = oid["gpu"]
    pbc = oid["pbc"]
    bias = oid["bias"]
    ortho = oid["ortho"]
    ddec6 = oid["ddec6"]
    pseudoSet = oid["pseudoSet"]
    freeze_base = oid["freeze_base"]
    freeze_tol = oid["freeze_tol"]
    freeze_count = oid["freeze_count"]
    exclude_freeze_count = oid["exclude_freeze_count"]
    freeze_map = oid["freeze_map"]
    freeze_all_but_map = oid["freeze_all_but_map"]
    if exclude_freeze_count > freeze_count:
        raise ValueError(f"freeze_count ({freeze_count}) must be greater than exclude_freeze_count ({exclude_freeze_count})")
    os.chdir(work_dir)
    opt_dir = opj(work_dir, "ion_opt/")
    lat_dir = opj(work_dir, "lat_opt/")
    sp_dir = opj(work_dir, "sp/")
    vib_dir = opj(work_dir, "vib/")
    opt_log = get_log_fn(work_dir, "sp_and_vib", False, restart=restart)
    opt_log(f"Given opt_input: {oid}")
    apply_freeze_func = get_apply_freeze_func(freeze_base, freeze_tol, freeze_count, None, exclude_freeze_count, freeze_map=freeze_map, freeze_all_but_map=freeze_all_but_map, log_fn=opt_log)
    structure = check_structure(structure, work_dir, log_fn=opt_log)
    atoms, restart = get_atoms(structure, True, work_dir, opt_dir, lat_dir, lat_iters, use_jdft, log_fn=opt_log)
    exe_cmd = get_exe_cmd(gpu, opt_log, use_srun=not debug)
    cmds_vib = get_cmds_dict(work_dir, ref_struct=structure, log_fn=opt_log, pbc=pbc, bias=bias, inputs_name="inputs_vib")
    cmds_vib = cmds_dict_to_list(cmds_vib)
    cmds_sp = get_cmds_dict(work_dir, ref_struct=structure, log_fn=opt_log, pbc=pbc, bias=bias, inputs_name="inputs_sp")
    cmds_sp = cmds_dict_to_list(cmds_sp)
    opt_log(f"Setting {structure} to atoms object")
    atoms.pbc = pbc
    cmds_sp = add_cohp_cmds(cmds_sp, ortho=ortho)
    if ddec6:
        cmds_sp = add_elec_density_dump(cmds_sp)
    base_infile_sp = cmds_list_to_infile(cmds_sp)
    base_infile_vib = cmds_list_to_infile(cmds_vib)
    sp_infile = get_ionic_opt_cmds_infile(base_infile_sp, ion_iters=0, use_jdft=use_jdft)
    vib_infile = get_ionic_opt_cmds_infile(base_infile_vib, ion_iters=0, use_jdft=use_jdft)
    if not "vibrations" in vib_infile:
        vib_infile.read_line("vibrations dumpK yes translationSym no useConstraints yes")
    get_arb_calc = lambda root, cmds: _get_calc_new(exe_cmd, cmds, root, pseudoSet=pseudoSet, debug=debug, log_fn=opt_log, ignore_cache_for_aimd=True, use_cart=oid["use_cart"])
    get_sp_calc = lambda root: get_arb_calc(root, sp_infile)
    get_vib_calc = lambda root: get_arb_calc(root, vib_infile)
    sp_finished = is_finished(sp_dir)
    vib_finished = is_finished(vib_dir)
    if not vib_finished:
        opt_log(f"Running vibrational calculation in {vib_dir}")
        run_ion_opt(atoms, vib_dir, get_vib_calc, apply_freeze_func, log_fn=opt_log)
    # if not sp_finished:
    #     opt_log(f"Running single point calculation in {sp_dir}")
    #     run_ion_opt(atoms, sp_dir, get_sp_calc, apply_freeze_func, log_fn=opt_log)
    if ddec6:
        opt_log(f"Running DDEC6 analysis in {opt_dir}")
        try:
            run_ddec6(sp_dir)
        except Exception as e:
            if ope(opj(sp_dir, "jdftx_run")):
                opt_log(f"Error running DDEC6: {e}, tryin again in {opj(sp_dir, 'jdftx_run')}")
                run_ddec6(opj(sp_dir, "jdftx_run"))
    

from sys import exc_info

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)

