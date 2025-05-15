from pathlib import Path
import os
from os.path import join as opj, exists as ope
from os import mkdir, getcwd,  chdir
from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from pymatgen.io.jdftx.inputs import JDFTXInfile
from helpers.calc_helpers import get_exe_cmd, _get_calc_new
from helpers.generic_helpers import (
    dump_template_input, fix_work_dir, check_submit,
    get_log_file_name,get_log_fn, log_def,
    get_inputs_list, cmds_list_to_infile, get_cmds_dict, cmds_dict_to_list,
    read_pbc_val, get_ref_struct, get_apply_freeze_func,
    _write_contcar
    )




neb_template = [
    "kval: 0.1 # Spring constant for band forces in NEB step",
    "neb method: eb # for autoneb, can be eb, aseneb, or improvedtangent. For custom images, can also be spline, tangent",
    "restart: True",
    "max_steps: 100 # max number of steps for bound optimization",
    "fmax: 0.025 # fmax perameter for both neb and scan opt",
    "pbc: True, true, true # which lattice vectors to impose periodic boundary conditions on",
    "relax: start, end # start optimizes given structure without frozen bond before scanning bond, end ",
    "gpu: True",
    "ase: True # Use ASE for optimization (good for making sure bounds are fulling optimized)",
    "climbing image: True",
    "images: 5 # Number of images provided",
    "bias: No_bias"
    ]


def read_neb_inputs(fname="neb_input"):
    """ Reads
    :param fname:
    :return:
    """
    if not Path(fname).exists():
        dump_template_input(fname, neb_template, getcwd())
        raise ValueError(f"No se neb input supplied")
    nid = {}
    nid["neb_method"] = "eb"
    nid["restart"] = True
    nid["max_steps"] = 100
    nid["step1_max_steps"] = 100
    nid["ci_max_steps"] = 100
    nid["fmax"] = 0.01
    nid["work_dir"] = None
    nid["relax_start"] = False
    nid["relax_end"] = False
    nid["pbc"] = [True, True, True]
    nid["gpu"] = False
    nid["pseudoSet"] = "GBRV"
    nid["freeze_base"] = False
    nid["freeze_tol"] = 0.
    nid["freeze_count"] = 0
    nid["freeze_idcs"] = None
    nid["exclude_freeze_count"] = 0
    nid["ci"] = True
    nid["images"] = 5
    nid["interp_method"] = "linear"
    nid["bias"] = "No_bias"
    nid["k"] = 0.1
    nid["ase"] = True
    inputs = get_inputs_list(fname, auto_lower=False)
    for input in inputs:
        key, val = input[0], input[1]
        if ("freeze" in key):
            if ("base" in key):
                nid["freeze_base"] = "true" in val.lower()
            elif ("tol" in key):
                nid["freeze_tol"] = float(val)
            elif ("count" in key):
                if "exclude" in key:
                    nid["exclude_freeze_count"] = int(val)
                else:
                    nid["freeze_count"] = int(val)
            elif ("idcs" in key):
                nid["freeze_idcs"] = [int(i) for i in val.split()]
        if "pseudo" in key:
            nid["pseudoSet"] = val.strip()
        if "gpu" in key:
            nid["gpu"] = "true" in val.lower()
        if "restart" in key:
            nid["restart"] = "true" in val.lower()
        if "work" in key:
            nid["work_dir"] = val.strip()
        if ("method" in key):
            if ("neb" in key):
                nid["neb_method"] = val.strip()
            elif ("interp" in key):
                nid["interp_method"] = val.strip()
        if key.lower() == "kval":
            nid["k"] = float(val.strip())
        if "max" in key:
            if "steps" in key:
                if "step1" in key:
                    nid["step1_max_steps"] = int(val.strip())
                elif "ci" in key:
                    nid["ci_max_steps"] = int(val.strip())
                else:
                    nid["max_steps"] = int(val.strip())
            elif ("force" in key) or ("fmax" in key):
                nid["fmax"] = float(val.strip())
        if "pbc" in key:
            nid["pbc"] = read_pbc_val(val)
        if "relax" in key:
            if "start" in val:
                nid["relax_start"] = True
            if "end" in val:
                nid["relax_end"] = True
        if "climbing" in key:
            if "image" in key:
                nid["ci"] = "true" in val.lower()
        if "images" in key:
            nid["images"] = int(val.strip())
            # if "start" in key:
            #     nid["images_start"] = int(val)
            # elif "par" in key:
            #     nid["images_par"] = int(val)
            # elif "max" in key:
            #     nid["images_max"] = int(val)
        if "struc" in key:
            if "start" in key:
                nid["start_struc"] = val.strip()
            elif "end" in key:
                nid["end_struc"] = val.strip()
        if "bias" in key:
            nid["bias"] = val.strip()
        if key.lower() == "kval":
            nid["k"] = float(val.strip())
        if key.strip().lower() == "ase":
            nid["ase"] = "true" in val.lower()
    nid["work_dir"] = fix_work_dir(nid["work_dir"])
    return nid

def get_initial_image_atoms(work_dir, struc_prefix, i, log_fn=log_def):
    """
    Get the initial image atoms from the work directory
    :param work_dir:
    :param struc_prefix:
    :param i:
    :return:
    """
    struc = f"{struc_prefix}{i}"
    struc_path = opj(work_dir, struc)
    atoms = None
    if not ope(struc_path):
        struc_path = opj(work_dir, f"{struc_prefix}{i}.gjf")
        if not ope(struc_path):
            return None
        log_fn(f"Reading initial image atoms from {struc_path}")
        atoms = read(struc_path, format="gaussian-in")
    else:
        log_fn(f"Reading initial image atoms from {struc_path}")
        atoms = read(struc_path, format="vasp")
    return atoms

def get_initial_image_atoms_list(work_dir, nimg_start, struc_prefix, log_fn=log_def, interp_method="idpp"):
    initial_image_atoms_list = []
    for i in range(nimg_start):
        initial_image_atoms_list.append(get_initial_image_atoms(work_dir, struc_prefix, i, log_fn=log_fn))
    if None in initial_image_atoms_list:
        log_fn(f"Missing initial image atoms, interpolating")
        initial_image_atoms_list = interpolate_missing_images(initial_image_atoms_list, interp_method, log_fn=log_fn)
    return initial_image_atoms_list


def interpolate_missing_images(_atoms_list, inter_method_str, log_fn=log_def):
    missing_idcs = [i for i in range(len(_atoms_list)) if _atoms_list[i] is None]
    atoms_list = []
    for i, _atoms in enumerate(_atoms_list):
        if _atoms is None:
            if i == 0:
                raise ValueError(f"Missing image {i} is the first image")
            __atoms = atoms_list[-1].copy()
            __atoms.calc = None
            atoms_list.append(__atoms)
        else:
            atoms_list.append(_atoms.copy())
    log_fn(f"Interpolating missing images {missing_idcs}")
    missing_runs = []
    missing_run = None
    for i in range(len(atoms_list)):
        if i in missing_idcs:
            if missing_run is None:
                missing_run = [i]
            else:
                missing_run.append(i)
        else:
            if missing_run is not None:
                missing_runs.append(missing_run)
                missing_run = None
    for missing_run in missing_runs:
        start = missing_run[0] - 1
        end = missing_run[-1] + 2
        atoms_sublist = atoms_list[start:end]
        tmp_neb = NEB(atoms_sublist)
        tmp_neb.interpolate(apply_constraint=True, method=inter_method_str)
        for i, idx in enumerate(missing_run):
            atoms_list[idx].set_positions(tmp_neb.images[i].get_positions())
    return atoms_list


def run_relax(
        work_dir, atoms, calc_fn, name,
        apply_freeze_func=None, 
        use_ase=False, fmax=0.01, max_steps=100,
        log_fn=log_def, log_file_path=None
        ):
    if not name is None:
        relax_dir = opj(work_dir, name)
    else:
        relax_dir = work_dir
    if not ope(relax_dir):
        mkdir(relax_dir)
    if not apply_freeze_func is None:
        atoms = apply_freeze_func(atoms)
    atoms.calc = calc_fn(relax_dir)
    if use_ase:
        if log_file_path is None:
            log_file_path = get_log_file_name(work_dir, "neb")
        dyn = FIRE(atoms, logfile=log_file_path, dt=0.01)
        dyn.run(fmax=fmax, steps=max_steps)
    else:
        atoms.get_potential_energy()
    return atoms


def relax_bounds(
        relax_start, relax_end, work_dir, start_atoms, end_atoms, restart,
        base_infile, get_arb_calc,
        fmax=0.01, max_steps=100,
        use_ase=False,
        log_fn=log_def, log_file_path=None):
    ion_infile = base_infile.copy()
    ion_infile["ionic-minimize"] = f"nIterations {max_steps}"
    get_ion_calc = lambda root: get_arb_calc(root, ion_infile)
    if relax_start:
        if Path(Path(work_dir) / "relax_start" / "CONTCAR").exists():
            start_atoms = read(opj(work_dir, opj("relax_start", "CONTCAR")), format="vasp")
        #if (not restart) or (not Path(Path(work_dir) / "relax_start" / "CONTCAR").exists()):
        relax_start_dir = opj(work_dir, "relax_start/")
        Path(relax_start_dir).mkdir(parents=True, exist_ok=True)
        start_atoms = run_relax(
            relax_start_dir, start_atoms, get_ion_calc, None,
            apply_freeze_func=None, 
            use_ase=use_ase, fmax=fmax, max_steps=max_steps,
            log_fn=log_def, log_file_path=log_file_path,
        )
        write(opj(work_dir, "relax_start", "CONTCAR"), start_atoms, format="vasp")
        # write(opj(work_dir, "relax_start", "CONTCAR"), start_atoms, format="vasp")
        # else:
        #     start_atoms = read(opj(work_dir, opj("relax_start", "CONTCAR")), format="vasp")
    if Path(Path(work_dir) / "relax_start" / "CONTCAR").exists():
        start_atoms = read(opj(work_dir, opj("relax_start", "CONTCAR")), format="vasp")
    if relax_end:
        if Path(Path(work_dir) / "relax_end" / "CONTCAR").exists():
            end_atoms = read(opj(work_dir, opj("relax_end", "CONTCAR")), format="vasp")
        #if (not restart) or (not ope(opj(work_dir, "relax_end", "CONTCAR"))):
        relax_end_dir = opj(work_dir, "relax_end/")
        Path(relax_end_dir).mkdir(parents=True, exist_ok=True)
        end_atoms = run_relax(
            relax_end_dir, end_atoms, get_ion_calc, None,
            apply_freeze_func=None, 
            use_ase=use_ase, fmax=fmax, max_steps=max_steps,
            log_fn=log_def, log_file_path=log_file_path,
        )
        write(opj(work_dir, "relax_end", "CONTCAR"), end_atoms, format="vasp")
        # else:
        #     end_atoms = read(opj(work_dir, opj("relax_end", "CONTCAR")), format="vasp")
    if Path(Path(work_dir) / "relax_end" / "CONTCAR").exists():
        end_atoms = read(opj(work_dir, opj("relax_end", "CONTCAR")), format="vasp")
    return start_atoms, end_atoms


def setup_img_dirs(
        work_dir, neb_dir, atoms_list
):
    _atoms_list = []
    for i, atoms in enumerate(atoms_list):
        img_dir = opj(neb_dir, f"{i}")
        if not ope(img_dir):
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            write(opj(img_dir, "POSCAR"), atoms, format="vasp")
            _atoms_list.append(atoms)
        elif ope(opj(img_dir, "CONTCAR")):
            _atoms = read(opj(img_dir, "CONTCAR"), format="vasp")
            _atoms_list.append(_atoms)
        elif ope(opj(img_dir, "POSCAR")):
            _atoms = read(opj(img_dir, "POSCAR"), format="vasp")
            _atoms_list.append(_atoms)
        else:
            write(opj(img_dir, "POSCAR"), atoms, format="vasp")
            _atoms_list.append(atoms)
    return _atoms_list


def setup_neb(
        work_dir: str, neb_dir: str, atoms_list: list[Atoms], base_infile: JDFTXInfile,
        get_arb_calc, use_ci: bool, k: float, neb_method: str, logfile: str):
    atoms_list = setup_img_dirs(work_dir, neb_dir, atoms_list)
    sp_infile = base_infile.copy()
    sp_infile["ionic-minimize"] = f"nIterations 0"
    get_sp_calc = lambda root: get_arb_calc(root, sp_infile)
    for i, atoms in enumerate(atoms_list):
        img_dir = opj(neb_dir, f"{i}/")
        atoms.calc = get_sp_calc(img_dir)
    neb = NEB(atoms_list, parallel=False, climb=use_ci, k=k, method=neb_method)
    dyn = FIRE(neb, logfile=logfile)
    traj = Trajectory(opj(neb_dir, "neb.traj"), 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj.write, interval=1)
    for i, img in enumerate(atoms_list):
        dyn.attach(lambda img, img_dir: _write_contcar(img, img_dir),
                   interval=1, img_dir=str(Path(neb_dir) / f"{i}/"), img=img)
    return dyn
        

    
    

def main(debug=False):
    nid = read_neb_inputs()
    restart = nid["restart"]
    gpu = nid["gpu"]
    pseudoSet = nid["pseudoSet"]
    pbc = nid["pbc"]
    nimg_start = nid["images"]
    # nimg_par = nid["images_par"]
    # nimg_max = nid["images_max"]
    use_ci = nid["ci"]
    k = nid["k"]
    neb_method = nid["neb_method"]
    interp_method = nid["interp_method"]
    max_steps = nid["max_steps"]
    # step1_max_steps = nid["step1_max_steps"]
    # ci_max_steps = nid["ci_max_steps"]
    fmax = nid["fmax"]
    bias = nid["bias"]
    work_dir = nid["work_dir"]
    freeze_base = nid["freeze_base"]
    freeze_tol = nid["freeze_tol"]
    freeze_count = nid["freeze_count"]
    freeze_idcs = nid["freeze_idcs"]
    exclude_freeze_count = nid["exclude_freeze_count"]
    relax_start = nid["relax_start"]
    relax_end = nid["relax_end"]
    ase = nid["ase"]
    chdir(work_dir)
    neb_log = get_log_fn(work_dir, "neb", debug, restart=restart)
    check_submit(gpu, os.getcwd(), "neb", log_fn=neb_log)
    neb_log_file = get_log_file_name(work_dir, "neb")
    apply_freeze_func = get_apply_freeze_func(
        freeze_base, freeze_tol, freeze_count, freeze_idcs, exclude_freeze_count,
        log_fn=neb_log
        )
    struc_prefix = "POSCAR_"
    ref_struc = get_ref_struct(work_dir, struc_prefix)
    cmds = get_cmds_dict(work_dir, ref_struct=ref_struc, log_fn=neb_log, pbc=pbc, bias=bias)
    cmds = cmds_dict_to_list(cmds)
    base_infile = cmds_list_to_infile(cmds)
    exe_cmd = get_exe_cmd(gpu, neb_log, use_srun=not debug)
    get_arb_calc = lambda root, cmds: _get_calc_new(exe_cmd, cmds, root, pseudoSet=pseudoSet, debug=debug, log_fn=neb_log)
    neb_log("Beginning NEB setup")
    neb_dir = opj(work_dir, "neb")
    Path(neb_dir).mkdir(parents=True, exist_ok=True)
    initial_images = get_initial_image_atoms_list(work_dir, nimg_start, struc_prefix, log_fn=neb_log, interp_method=interp_method)
    start_atoms, end_atoms = relax_bounds(
        relax_start, relax_end, work_dir, initial_images[0], initial_images[-1], restart,
        base_infile, get_arb_calc,
        fmax=fmax, max_steps=max_steps,
        use_ase=ase,
        log_fn=neb_log, log_file_path=get_log_file_name(work_dir, "neb"),
    )
    initial_images[0] = start_atoms
    initial_images[-1] = end_atoms
    dyn = setup_neb(
        work_dir, neb_dir, initial_images,
        base_infile, get_arb_calc, use_ci, k, neb_method,
        neb_log_file
    )
    dyn.run(fmax=fmax, steps=max_steps)



from sys import exc_info, stderr

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)
