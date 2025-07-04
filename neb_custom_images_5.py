from pathlib import Path
import os
from os.path import join as opj, exists as ope
from os import mkdir, getcwd,  chdir
from ase import Atoms, Atom
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
    _write_contcar, add_cohp_cmds, add_elec_density_dump
    )
from scripts.run_ddec6_v3 import main as run_ddec6
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.ase import AseAtomsAdaptor



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
    "jdftx: True # Use JDFTx for optimization (If ASE also True, runs between optimization steps for boundaries)",
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
    nid["freeze_idcs"] = []
    nid["exclude_freeze_count"] = 0
    nid["ci"] = True
    nid["images"] = 5
    nid["interp_method"] = "idpp"
    nid["bias"] = "No_bias"
    nid["k"] = 0.1
    nid["ase"] = True
    nid["jdftx"] = True
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
                nid["freeze_idcs"] = [int(i.rstrip(",")) for i in val.split()]
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
            _val = val.strip()
            if len(_val.split()) == 1:
                nid["k"] = float(_val)
            else:
                nid["k"] = [float(v.rstrip(",")) for v in _val.split()]
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
        if key.strip().lower() == "ase":
            nid["ase"] = "true" in val.lower()
        if "jdft" in key.lower():
            nid["jdftx"] = "true" in val.lower()
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

def get_initial_image_atoms_list(work_dir, nimg_start, struc_prefix, log_fn=log_def):
    initial_image_atoms_list = []
    for i in range(nimg_start):
        initial_image_atoms_list.append(get_initial_image_atoms(work_dir, struc_prefix, i, log_fn=log_fn))
    if initial_image_atoms_list[0] is None:
        log_fn(f"First initial image atoms not found in {work_dir} (should be named {struc_prefix}0)")
        raise ValueError(f"First initial image atoms not found in {work_dir} (should be named {struc_prefix}0)")
    if initial_image_atoms_list[-1] is None:
        log_fn(f"Last initial image atoms not found in {work_dir} (should be named {struc_prefix}{nimg_start-1})")
        raise ValueError(f"Last initial image atoms not found in {work_dir} (should be named {struc_prefix}{nimg_start-1})")
    return initial_image_atoms_list
    


def check_initial_atoms_list(initial_image_atoms_list, log_fn=log_def, interp_method="idpp"):
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
        for i, idx in enumerate(missing_run): # Missing run only contains the missing images
            # while tmp_neb contains the bounds as well, so [i+1] is needed to skip the first image
            atoms_list[idx].set_positions(tmp_neb.images[i+1].get_positions())
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
        dyn = FIRE(atoms, logfile=log_file_path)
        dyn.run(fmax=fmax, steps=max_steps)
    else:
        atoms.get_potential_energy()
    return atoms



def get_bounds_atoms(bound_relax_dir, template_atoms, log_fn=log_def):
    outpath = Path(bound_relax_dir) / "jdftx_run" / "out"
    contcar = Path(bound_relax_dir) / "CONTCAR"
    traj = Path(bound_relax_dir) / "_restart.traj"
    if all([not contcar.exists(), not traj.exists(), not outpath.exists()]):
        log_fn(f"Neither out, CONTCAR, nor _restart.traj exist in {bound_relax_dir}")
        return template_atoms
    elif all([contcar.exists(), traj.exists()]):
        if contcar.stat().st_mtime > traj.stat().st_mtime:
            log_fn(f"Reading CONTCAR from {bound_relax_dir} since it is newer than _restart.traj")
            template_atoms = read(contcar, format="vasp")
        else:
            log_fn(f"Reading _restart.traj from {bound_relax_dir} since it is newer than CONTCAR")
            template_atoms = read(traj)
    elif contcar.exists():
        log_fn(f"Reading CONTCAR from {bound_relax_dir}")
        template_atoms = read(contcar, format="vasp")
    elif traj.exists():
        log_fn(f"Reading _restart.traj from {bound_relax_dir}")
        template_atoms = read(traj)
    elif outpath.exists():
        outfile = JDFTXOutfile.from_file(outpath)
        template_atoms = AseAtomsAdaptor.get_atoms(outfile.structure)
    return template_atoms


def relax_bounds(
        relax_start, relax_end, work_dir, start_atoms, end_atoms, restart,
        base_infile, get_arb_calc,
        fmax=0.01, max_steps=100,
        use_ase=False, use_jdftx=True,
        apply_freeze_func=None,
        log_fn=log_def, log_file_path=None):
    if ((not use_ase) and (not use_jdftx)):
        use_jdftx = True
    ion_infile = base_infile.copy()
    if use_jdftx:
        ion_infile["ionic-minimize"] = f"nIterations {max_steps}"
    else:
        ion_infile["ionic-minimize"] = f"nIterations 0"
    get_ion_calc = lambda root: get_arb_calc(root, ion_infile)
    relax_start_dir = opj(work_dir, "relax_start/")
    start_atoms = get_bounds_atoms(relax_start_dir, start_atoms)
    relax_end_dir = opj(work_dir, "relax_end/")
    end_atoms = get_bounds_atoms(relax_end_dir, end_atoms)
    if relax_start:
        start_atoms = get_bounds_atoms(relax_start_dir, start_atoms)
        Path(relax_start_dir).mkdir(parents=True, exist_ok=True)
        start_atoms = run_relax(
            relax_start_dir, start_atoms, get_ion_calc, None,
            apply_freeze_func=apply_freeze_func, 
            use_ase=use_ase, fmax=fmax, max_steps=max_steps,
            log_fn=log_def, log_file_path=log_file_path,
        )
        write(opj(work_dir, "relax_start", "CONTCAR"), start_atoms, format="vasp")
    if relax_end:
        end_atoms = get_bounds_atoms(relax_end_dir, end_atoms)
        Path(relax_end_dir).mkdir(parents=True, exist_ok=True)
        end_atoms = run_relax(
            relax_end_dir, end_atoms, get_ion_calc, None,
            apply_freeze_func=apply_freeze_func, 
            use_ase=use_ase, fmax=fmax, max_steps=max_steps,
            log_fn=log_def, log_file_path=log_file_path,
        )
        write(opj(work_dir, "relax_end", "CONTCAR"), end_atoms, format="vasp")
    return start_atoms, end_atoms


def setup_img_dirs(
        work_dir, neb_dir, atoms_list
):
   #  _atoms_list = []
    for i, atoms in enumerate(atoms_list):
        img_dir = opj(neb_dir, f"{i}")
        if not ope(img_dir):
            Path(img_dir).mkdir(parents=True, exist_ok=True)
    #         write(opj(img_dir, "POSCAR"), atoms, format="vasp")
    #         _atoms_list.append(atoms)
    #     elif ope(opj(img_dir, "CONTCAR")):
    #         _atoms = read(opj(img_dir, "CONTCAR"), format="vasp")
    #         _atoms_list.append(_atoms)
    #     elif ope(opj(img_dir, "POSCAR")):
    #         _atoms = read(opj(img_dir, "POSCAR"), format="vasp")
    #         _atoms_list.append(_atoms)
    #     else:
    #         write(opj(img_dir, "POSCAR"), atoms, format="vasp")
    #         _atoms_list.append(atoms)
    # return _atoms_list


def set_atoms_list_from_traj(neb_dir, atoms_list):
    nimg = len(atoms_list)
    _atoms_list = None
    trajfile = opj(neb_dir, "neb.traj")
    try:
        _atoms_list = Trajectory(trajfile)[-nimg:]
        # Prevent from setting Atom objects in the atoms_list
        assert all([not isinstance(atoms, Atom) for atoms in _atoms_list])
    except:
        pass
    if not _atoms_list is None:
        for i, atoms in enumerate(_atoms_list):
            if not i in [0, nimg-1]:
                # atoms.calc = None
                atoms_list[i] = atoms
    return atoms_list


def setup_neb(
        work_dir: str, neb_dir: str, atoms_list: list[Atoms], base_infile: JDFTXInfile,
        get_arb_calc, use_ci: bool, k: float | list[float], neb_method: str, logfile: str, apply_freeze_func=None):
    setup_img_dirs(work_dir, neb_dir, atoms_list)
    # Using the trajectory is more robust as it won't allow initializing a partially updated set of images
    atoms_list = set_atoms_list_from_traj(neb_dir, atoms_list)
    sp_infile = base_infile.copy()
    sp_infile["ionic-minimize"] = f"nIterations 0"
    get_sp_calc = lambda root: get_arb_calc(root, sp_infile)
    for i, atoms in enumerate(atoms_list):
        img_dir = opj(neb_dir, f"{i}/")
        atoms = apply_freeze_func(atoms)
        atoms.calc = get_sp_calc(img_dir)
    neb = NEB(atoms_list, parallel=False, climb=use_ci, k=k, method=neb_method)
    dyn = FIRE(neb, logfile=logfile, restart=opj(neb_dir, "hessian.pckl"))
    tmode = 'a' if Path(opj(neb_dir, "neb.traj")).exists() else 'w'
    traj = Trajectory(opj(neb_dir, "neb.traj"), tmode, neb, properties=['energy', 'forces'])
    dyn.attach(traj.write, interval=1)
    for i, img in enumerate(atoms_list):
        dyn.attach(lambda img, img_dir: _write_contcar(img, img_dir),
                   interval=1, img_dir=str(Path(neb_dir) / f"{i}/"), img=img)
    return dyn


def setuprun_neb_post_anl(
        work_dir: str, neb_dir: str, atoms_list, neb_anl_dir: str, wdump_infile: JDFTXInfile,
        get_arb_calc, use_ci: bool, k: float | list[float], neb_method: str, logfile: str, apply_freeze_func=None):
    setup_img_dirs(work_dir, neb_dir, atoms_list)
    # Using the trajectory is more robust as it won't allow initializing a partially updated set of images
    atoms_list = set_atoms_list_from_traj(neb_dir, atoms_list)
    sp_infile = wdump_infile.copy()
    sp_infile["ionic-minimize"] = f"nIterations 0"
    get_sp_calc = lambda root: get_arb_calc(root, sp_infile)
    for i, atoms in enumerate(atoms_list):
        img_dir = opj(neb_anl_dir, f"{i}/")
        atoms = apply_freeze_func(atoms)
        atoms.calc = get_sp_calc(img_dir)
    # Ensure bounds are ran before realizing band is converged by forcing method as spline
    neb = NEB(atoms_list, parallel=False, climb=use_ci, k=k, method="spline")
    dyn = FIRE(neb, logfile=logfile, restart=opj(neb_anl_dir, "hessian.pckl"))
    tmode = 'w'
    traj = Trajectory(opj(neb_anl_dir, "neb.traj"), tmode, neb, properties=['energy', 'forces'])
    dyn.attach(traj.write, interval=1)
    for i, img in enumerate(atoms_list):
        # dyn.attach(lambda img, img_dir: _write_contcar(img, img_dir),
        #            interval=1, img_dir=str(Path(neb_anl_dir) / f"{i}/"), img=img)
        dyn.attach(lambda img_dir_run_dir: run_ddec6(img_dir_run_dir),
                   interval=1, img_dir_run_dir=str(Path(neb_anl_dir) / f"{i}/jdftx_run/"))
    cwd = getcwd()
    dyn.attach(
        lambda: chdir(cwd), interval=1
        )  # Ensure we return to the original working directory after each step
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
    jdftx = nid["jdftx"]
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
    if ope(opj(work_dir, "anl_inputs")):
        anl_cmds = get_cmds_dict(work_dir, ref_struct=ref_struc, log_fn=neb_log, pbc=pbc, bias=bias, inputs_name="anl_inputs")
        anl_cmds = cmds_dict_to_list(anl_cmds)
    else:
        print("No anl_inputs found, using default commands")
        anl_cmds = cmds_dict_to_list(get_cmds_dict(work_dir, ref_struct=ref_struc, log_fn=neb_log, pbc=pbc, bias=bias))
    wdump_cmds = add_cohp_cmds(anl_cmds.copy())
    wdump_cmds = add_elec_density_dump(wdump_cmds)
    base_infile = cmds_list_to_infile(cmds)
    wdump_infile = cmds_list_to_infile(wdump_cmds)
    exe_cmd = get_exe_cmd(gpu, neb_log, use_srun=not debug)
    get_arb_calc = lambda root, cmds: _get_calc_new(exe_cmd, cmds, root, pseudoSet=pseudoSet, debug=debug, log_fn=neb_log)
    neb_log("Beginning NEB setup")
    neb_dir = opj(work_dir, "neb")
    Path(neb_dir).mkdir(parents=True, exist_ok=True)
    initial_images = get_initial_image_atoms_list(work_dir, nimg_start, struc_prefix, log_fn=neb_log)
    # This is run regardless of whether boundary relaxation is requested or not, as if there are boundary
    # relaxation calculations found, they will be used as the boundaries
    start_atoms, end_atoms = relax_bounds(
        relax_start, relax_end, work_dir, initial_images[0], initial_images[-1], restart,
        base_infile, get_arb_calc,
        fmax=fmax, max_steps=max_steps,
        use_ase=ase, use_jdftx=jdftx,
        apply_freeze_func=apply_freeze_func,
        log_fn=neb_log, log_file_path=get_log_file_name(work_dir, "neb"),
    )
    initial_images[0] = start_atoms
    initial_images[-1] = end_atoms
    initial_images = check_initial_atoms_list(initial_images, interp_method=interp_method, log_fn=neb_log)
    dyn = setup_neb(
        work_dir, neb_dir, initial_images,
        base_infile, get_arb_calc, use_ci, k, neb_method,
        neb_log_file, apply_freeze_func=apply_freeze_func,
    )
    dyn.run(fmax=fmax, steps=max_steps)
    neb_anl_dir = opj(work_dir, "neb_anl")
    Path(neb_anl_dir).mkdir(parents=True, exist_ok=True)
    anl_dyn = setuprun_neb_post_anl(
        work_dir, neb_dir, initial_images, neb_anl_dir, wdump_infile,
        get_arb_calc, use_ci, k, neb_method,
        neb_log_file, apply_freeze_func=apply_freeze_func
    )
    anl_dyn.run(fmax=fmax*100, steps=0)

from sys import exc_info, stderr

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)
