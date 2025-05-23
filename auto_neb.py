from pathlib import Path
from helpers.generic_helpers import get_inputs_list, read_pbc_val, fix_work_dir, dump_template_input, get_ref_struct, get_apply_freeze_func
from os import getcwd, chdir
import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from os.path import join as opj, exists as ope, isdir,  basename
from os import rename
from os import mkdir, getcwd,  chdir

import ase.parallel as mpi
from ase.mep import NEB, AutoNEB
from datetime import datetime
#from ase.neb import NEB
from ase.mep import NEB
import numpy as np
from opt import run_ion_opt
import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from os.path import join as opj, exists as ope
from os import mkdir, getcwd,  chdir
import numpy as np
from neb import setup_img_dirs
from pathlib import Path
from pymatgen.io.jdftx.inputs import JDFTXInfile
from helpers.generic_helpers import get_int_dirs, copy_state_files, get_cmds_dict, get_int_dirs_indices, \
    get_atoms_list_from_out, get_do_cell, get_atoms
from helpers.generic_helpers import get_int_dirs, copy_state_files, get_cmds_dict, get_int_dirs_indices, \
    get_atoms_list_from_out, get_do_cell, get_atoms, get_ionic_opt_cmds_list
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, _write_contcar, optimizer
from helpers.generic_helpers import dump_template_input, get_log_fn, copy_file, log_def, cmds_list_to_infile

from helpers.generic_helpers import _write_opt_iolog, check_for_restart, get_nrg, _write_img_opt_iolog
from helpers.generic_helpers import remove_dir_recursive, get_ionic_opt_cmds_dict, check_submit, cmds_dict_to_list, add_freeze_surf_base_constraint
from helpers.generic_helpers import get_int_dirs, copy_state_files, get_cmds_dict, get_int_dirs_indices, \
    get_atoms_list_from_out, get_do_cell, get_atoms
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, _write_contcar, optimizer, get_infile
from helpers.generic_helpers import dump_template_input, get_log_fn, log_def, get_log_file_name

from helpers.generic_helpers import _write_opt_iolog, check_for_restart
from helpers.generic_helpers import cmds_dict_to_list
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.calc_helpers import _get_calc, get_exe_cmd, _get_calc_new
from helpers.logx_helpers import _write_logx



neb_template = [
    "kval: 0.1 # Spring constant for band forces in NEB step",
    "neb method: eb # for autoneb, can be eb, aseneb, or improvedtangent",
    "restart: True",
    "max_steps: 100 # max number of steps for bound optimization",
    "step1_max_steps: 100 # max number of steps for pre-climbing image step",
    "ci_max_steps: 100 # max number of steps for climbing image step",
    "fmax: 0.025 # fmax perameter for both neb and scan opt",
    "pbc: True, true, true # which lattice vectors to impose periodic boundary conditions on",
    "relax: start, end # start optimizes given structure without frozen bond before scanning bond, end ",
    "gpu: True",
    "ase: True # Use ASE for optimization (good for making sure bounds are fulling optimized)",
    "climbing image: True",
    "images_start: 2 # Number of images provided",
    "images_par: 3 # Number of images ran in a single autoneb step",
    "# (Fewer allows autoneb to focus on more important images, but more allows for saved wavefunction files",
    "# to be be saved for the same image they were created for)",
    "images_max: 10",
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
    nid["provided_images"] = 10
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
            if "start" in key:
                nid["images_start"] = int(val)
            elif "par" in key:
                if "*" in val:
                    nid["images_par"] = None
                else:
                    nid["images_par"] = int(val)
            elif "max" in key:
                nid["images_max"] = int(val)
        if "struc" in key:
            if "start" in key:
                nid["start_struc"] = val.strip()
            elif "end" in key:
                nid["end_struc"] = val.strip()
        if "bias" in key:
            nid["bias"] = val.strip()
        if key.lower() == "kval":
            nid["k"] = float(val.strip())
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
            raise ValueError(f"Could not find initial image {i} in {work_dir} for prefix {struc_prefix}")
        log_fn(f"Reading initial image atoms from {struc_path}")
        atoms = read(struc_path, format="gaussian-in")
    else:
        log_fn(f"Reading initial image atoms from {struc_path}")
        atoms = read(struc_path, format="vasp")
    return atoms

def get_initial_image_atoms_list(work_dir, nimg_start, struc_prefix, log_fn=log_def):
    initial_image_atoms_list = []
    for i in range(nimg_start):
        initial_image_atoms_list.append(get_initial_image_atoms(work_dir, struc_prefix, i, log_fn=log_fn))
    return initial_image_atoms_list


def run_initial_images(
        work_dir, initial_images_dir, neb_dir, nimg_start, struc_prefix, 
        get_arb_calc, base_infile,
        relax_start, relax_end, fmax, max_steps, apply_freeze_func,
        use_ase=False,
        restart=False, log_fn=log_def, debug=False,
        ):
    atoms_list = get_initial_image_atoms_list(work_dir, nimg_start, struc_prefix, log_fn=log_fn)
    write_traj_paths = [str(Path(neb_dir) / f"j{i:03d}.traj") for i in range(nimg_start)]
    restart = restart and all([ope(p) for p in write_traj_paths])
    for i, atoms in enumerate(atoms_list):
        initial_images_dir_i = opj(initial_images_dir, f"{i}/")
        if not ope(initial_images_dir_i):
            restart = False
            _dir = Path(initial_images_dir_i)
            _dir.mkdir(parents=True, exist_ok=True)
        if not restart:
            use_infile = base_infile.copy()
            _use_ase = True
            # calc_fn = get_sp_calc
            if ((i == 0 and relax_start) or (i == nimg_start - 1 and relax_end)):
                use_infile["ionic-minimize"] = f"nIterations {max_steps}"
            else:
                use_infile["ionic-minimize"] = f"nIterations 0"
                _use_ase = False
            calc_fn = lambda root: get_arb_calc(root, use_infile)
            ran_atoms = run_relax(
                initial_images_dir_i, atoms, calc_fn, None,
                use_ase=(use_ase and _use_ase),
                fmax=fmax, max_steps=max_steps,
                apply_freeze_func=apply_freeze_func,
                log_fn=log_fn, log_file_path=get_log_file_name(work_dir, "neb")
                )
            write(str(Path(neb_dir) / f"j{i:03d}.traj"), ran_atoms, format="traj")


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
        dyn = FIRE(atoms, logfile=log_file_path)
        dyn.run(fmax=fmax, steps=max_steps)
    else:
        atoms.get_potential_energy()
    return atoms
    
    

def main(debug=False):
    nid = read_neb_inputs()
    restart = nid["restart"]
    gpu = nid["gpu"]
    pseudoSet = nid["pseudoSet"]
    pbc = nid["pbc"]
    nimg_start = nid["images_start"]
    nimg_par = nid["images_par"]
    nimg_max = nid["images_max"]
    use_ci = nid["ci"]
    k = nid["k"]
    neb_method = nid["neb_method"]
    interp_method = nid["interp_method"]
    max_steps = nid["max_steps"]
    step1_max_steps = nid["step1_max_steps"]
    ci_max_steps = nid["ci_max_steps"]
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
    initial_images_dir = opj(work_dir, "initial_images")
    neb_dir = opj(work_dir, "neb")
    Path(neb_dir).mkdir(parents=True, exist_ok=True)
    check_submit(gpu, os.getcwd(), "auto_neb", log_fn=neb_log)
    restart = run_initial_images(
        work_dir, initial_images_dir, neb_dir, nimg_start, struc_prefix, 
        get_arb_calc, base_infile,
        relax_start, relax_end, fmax, max_steps, apply_freeze_func,
        use_ase=ase,
        restart=restart, log_fn=neb_log, debug=debug,
        )
    use_infile = base_infile.copy()
    use_infile["ionic-minimize"] = f"nIterations 0"
    def attach_calculators(images):
        print(f"Attaching calculators on {len(images)} images")
        for i, image in enumerate(images):
            image.calc = get_arb_calc(str(Path(neb_dir) / f"{i}"), use_infile)
    nimg_par = set_nimg_par(nimg_par, neb_dir)
    neb = AutoNEB(
        attach_calculators,
        str(str(Path(neb_dir) / "j")),
        nimg_par,
        nimg_max,
        parallel=mpi.world.rank > 0,
        climb=use_ci,
        iter_folder=neb_dir,
        maxsteps=[step1_max_steps, ci_max_steps],
        method=neb_method,
        interpolate_method=interp_method
        )
    try:
        neb.run()
    except StopIteration as e:
        neb_log(f"NEB run stopped: {e}")
        neb_log("Cleaning up NEB directory")
        remove_dir_recursive(neb_dir)
        Path(neb_dir).mkdir(parents=True, exist_ok=True)
        run_initial_images(
            work_dir, initial_images_dir, neb_dir, nimg_start, struc_prefix, 
            get_arb_calc, base_infile,
            relax_start, relax_end, fmax, max_steps, apply_freeze_func,
            use_ase=ase,
            restart=True, log_fn=neb_log, debug=debug,
        )
        neb_log(f"Trying again")
        neb.run()


def set_nimg_par(nimg_par, neb_dir):
    if isinstance(nimg_par, int):
        return nimg_par
    elif nimg_par is None:
        ncur = get_cur_num_images(neb_dir)
        if ncur > 2:
            nimg_par = ncur
        else:
            nimg_par = 2
    else:
        nimg_par = int(nimg_par)
    return nimg_par

def get_cur_num_images(neb_dir):
    fs = os.listdir(neb_dir)
    fs = [f for f in fs if f.endswith(".traj")]
    fs = [f for f in fs if "iter" in f]
    image_fs = {}
    for f in fs:
        img_str = f.split("j")[1].split(".")[0]
        if img_str in image_fs:
            image_fs[img_str].append(f)
        else:
            image_fs[img_str] = [f]
    cur_num_images = len(image_fs)
    return cur_num_images
    
# def repair_auto_neb_dir(neb_dir):
#     fs = os.listdir(neb_dir)
#     fs = [f for f in fs if f.endswith(".traj")]
#     fs = [f for f in fs if "iter" in f]
#     image_fs = {}
#     for f in fs:
#         img_str = f.split("j")[1].split(".")[0]
#         if img_str in image_fs:
#             image_fs[img_str].append(f)
#         else:
#             image_fs[img_str] = [f]
#     for img_str, fs in image_fs.items():
#         iter_fs = [f.split("iter")[1].split(".traj")[0] for f in fs]
#         idcs = np.argsort([int(i) for i in iter_fs])
#         most_recent_f = fs[idcs[-1]]
#         copy_file(
#             opj(neb_dir, most_recent_f),
#             opj(neb_dir, f"j{img_str}.traj"),
#             )



from sys import exc_info, stderr

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)
