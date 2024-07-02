import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from os.path import join as opj, exists as ope, isdir,  basename
from os import mkdir, getcwd,  chdir
from ase.neb import NEB
from helpers.generic_helpers import get_int_dirs, copy_state_files, get_cmds_dict, get_int_dirs_indices, \
    get_atoms_list_from_out, get_do_cell, get_atoms
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, _write_contcar, optimizer
from helpers.generic_helpers import dump_template_input, get_log_fn, copy_file, log_def, add_freeze_surf_base_constraint
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.generic_helpers import _write_opt_iolog, check_for_restart, get_nrg, _write_img_opt_iolog
from helpers.generic_helpers import remove_dir_recursive, get_ionic_opt_cmds_dict, check_submit, cmds_dict_to_list, add_freeze_surf_base_constraint
from helpers.geom_helpers import get_property
from helpers.generic_helpers import death_by_nan, reset_atoms_death_by_nan, check_structure, _check_structure
from helpers.logx_helpers import write_scan_logx, out_to_logx, _write_logx, finished_logx, sp_logx
from helpers.generic_helpers import add_freeze_list_constraints, copy_best_state_files, get_atom_str
from helpers.se_neb_helpers import get_nrgs, has_max, check_poscar, neb_optimizer, count_scan_steps, _prep_input, setup_scan_dir, is_max
from helpers.schedule_helpers import write_autofill_schedule, j_steps_key, freeze_list_key, read_schedule_file, \
    get_step_list, energy_key, properties_key, get_prop_idcs_list, append_results_as_comments, \
    get_scan_steps_list_for_neb, get_neb_options, insert_finer_steps, write_schedule_to_text

neb_template = ["kval: 0.1 # Spring constant for band forces in NEB step",
                   "neb method: spline # idk, something about how forces are projected out / imposed",
                   "restart: True",
                   "max_steps: 100 # max number of steps for scan opts",
                   "fmax: 0.05 # fmax perameter for both neb and scan opt",
                   "pbc: True, true, false # which lattice vectors to impose periodic boundary conditions on",
                   "relax: start, end # start optimizes given structure without frozen bond before scanning bond, end ",
                   "gpu: True",
                   "climbing image: True",
                   "images: 10",
                   "start struc: POSCAR_start",
                "end struc: POSCAR_end",
                "bias: No_bias"]

debug = False
if debug:
    from os import environ
    environ["JDFTx_pseudo"] = "E:\\volD\\scratch_backup\\pseudopotentials"
    environ["JDFTx_GPU"] = "None"
    environ["JDFTx"] = "None"

def read_neb_inputs(fname="neb_input"):
    """ Reads
    :param fname:
    :return:
    """
    if not ope(fname):
        dump_template_input(fname, neb_template, getcwd())
        raise ValueError(f"No se neb input supplied: dumping template {fname}")
    nid = {}
    nid["k"] = 1.0
    nid["neb_method"] = "spline"
    nid["restart"] = True
    nid["max_steps"] = 100
    nid["fmax"] = 0.01
    nid["work_dir"] = None
    nid["relax_start"] = False
    nid["relax_end"] = False
    nid["pbc"] = [True, True, False]
    nid["gpu"] = False
    nid["pseudoSet"] = "GBRV"
    nid["freeze_base"] = False
    nid["freeze_tol"] = 0.
    nid["freeze_count"] = 0
    nid["ci"] = True
    nid["images"] = 10
    nid["interp_method"] = "linear"
    nid["bias"] = "No_bias"
    inputs = get_inputs_list(fname, auto_lower=False)
    for input in inputs:
        key, val = input[0], input[1]
        if ("freeze" in key):
            if ("base" in key):
                nid["freeze_base"] = "true" in val.lower()
            elif ("tol" in key):
                nid["freeze_tol"] = float(val)
            elif ("count" in key):
                nid["freeze_count"] = int(val)
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
            nid["kval"] = float(val.strip())
        if "max" in key:
            if "steps" in key:
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
            nid["images"] = int(val)
        if "struc" in key:
            if "start" in key:
                nid["start_struc"] = val.strip()
            elif "end" in key:
                nid["end_struc"] = val.strip()
        if "bias" in key:
            nid["bias"] = val.strip()
    nid["work_dir"] = fix_work_dir(nid["work_dir"])
    return nid


def parse_lookline(lookline):
    step_length = float(lookline[-1])
    scan_steps = int(lookline[-2])
    atom_idcs = []
    for idx in lookline[:-2]:
        atom_idcs.append(int(idx) - 1)
    return atom_idcs, scan_steps, step_length



def finished(dir_path):
    with open(opj(dir_path, f"finished_{basename(dir_path)}.txt"), "w") as f:
        f.write("done")


def is_done(dir_path):
    idx = int(basename(dir_path))
    return ope(opj(dir_path, f"finished_{idx}.txt"))


def get_restart_idx(restart_idx, scan_path, log_fn=log_def):
    if not restart_idx is None:
        log_fn(f"Restart index specified at {restart_idx}")
        return restart_idx
    else:
        restart_idx = 0
        if not ope(scan_path):
            mkdir(scan_path)
            return restart_idx
        else:
            int_dirs = get_int_dirs(scan_path)
            int_dirs_indices = get_int_dirs_indices(int_dirs)
            for i in range(len(int_dirs)):
                look_dir = int_dirs[int_dirs_indices[i]]
                if ope(look_dir):
                    if is_done(look_dir):
                        restart_idx = i+1
                    else:
                        return i
                else:
                    return i
            return restart_idx


def run_jdftx_opt(atoms_obj, root_path, log_fn=log_def):
    outfile = opj(root_path, "out")
    log_fn("JDFTx pre-optimization starting")
    atoms_obj.get_forces()
    log_fn("JDFTx pre-optimization finished")
    jdft_opt = opj(root_path, "jdftx_opt")
    if not ope(jdft_opt):
        mkdir(jdft_opt)
    out_to_logx(jdft_opt, outfile, log_fn=log_fn)
    new_atoms = get_atoms_list_from_out(outfile)[-1]
    atoms_obj.set_positions(new_atoms.positions)
    write(opj(jdft_opt, "CONTCAR"), atoms_obj, format="vasp")
    return atoms_obj


def run_opt_runner(atoms_obj, root_path, opter, log_fn = log_def, fmax=0.05, max_steps=100):
    dyn = optimizer(atoms_obj, root_path, opter)
    traj = Trajectory(opj(root_path, "opt.traj"), 'w', atoms_obj, properties=['energy', 'forces', 'charges'])
    logx = opj(root_path, "opt.logx")
    do_cell = get_do_cell(atoms_obj.pbc)
    dyn.attach(traj.write, interval=1)
    dyn.attach(lambda: _write_contcar(atoms_obj, root_path), interval=1)
    dyn.attach(lambda: _write_logx(atoms_obj, logx, do_cell=do_cell), interval=1)
    dyn.attach(lambda: _write_opt_iolog(atoms_obj, dyn, max_steps, log_fn), interval=1)
    log_fn("Optimization starting")
    log_fn(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
    finished_logx(atoms_obj, logx, dyn.nsteps, max_steps)
    sp_logx(atoms_obj, "sp.logx", do_cell=do_cell)
    finished(root_path)


def run_step_runner(atoms_obj, step_path, opter, get_calc_fn, j_steps, get_jdft_opt_calc_fn,
                    log_fn = log_def, fmax=0.05, max_steps=100):
    if j_steps > 0:
        atoms_obj.set_calculator(get_jdft_opt_calc_fn(step_path, j_steps))
        atoms_obj = run_jdftx_opt(atoms_obj, step_path, log_fn=log_fn)
    atoms_obj.set_calculator(get_calc_fn(step_path))
    run_opt_runner(atoms_obj, step_path, opter, log_fn=log_fn, fmax=fmax, max_steps=max_steps)



def read_instructions_run_step(instructions):
    freeze_list = instructions[freeze_list_key]
    j_steps = instructions[j_steps_key]
    return freeze_list, j_steps



def run_step(atoms_obj, step_path, instructions, get_jdft_opt_calc_fn, get_calc_fn, opter_ase_fn,
             fmax_float=0.1, max_steps_int=50, freeze_base=False, freeze_tol=1.0, log_fn=log_def, _failed_before_bool=False):
    freeze_list, j_steps = read_instructions_run_step(instructions)
    run_again = False
    add_freeze_list_constraints(atoms_obj, freeze_list, log_fn=log_fn)
    add_freeze_surf_base_constraint(atoms_obj, freeze_base=freeze_base, ztol=freeze_tol, log_fn=log_def)
    try:
        run_step_runner(atoms_obj, step_path, opter_ase_fn, get_calc_fn, j_steps, get_jdft_opt_calc_fn, log_fn=log_fn, fmax=fmax_float, max_steps=max_steps_int)
    except Exception as e:
        check_for_restart(e, _failed_before_bool, step_path, log_fn=log_fn)
        if death_by_nan(opj(step_path, "out"), log_def):
            atoms_obj = reset_atoms_death_by_nan(step_path, step_path)
            add_freeze_list_constraints(atoms_obj, freeze_list, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        run_step(atoms_obj, step_path, instructions, get_jdft_opt_calc_fn, get_calc_fn, opter_ase_fn,
                 fmax_float=fmax_float, max_steps_int=max_steps_int, log_fn=log_fn, _failed_before_bool=True)



def run_relax_opt(atoms_obj, opt_path, opter_ase_fn, get_calc_fn,
                  fmax_float=0.05, max_steps_int=100, log_fn=log_def, _failed_before_bool=False):
    atoms_obj.set_calculator(get_calc_fn(opt_path))
    run_again = False
    try:
        run_opt_runner(atoms_obj, opt_path, opter_ase_fn, fmax=fmax_float, max_steps=max_steps_int, log_fn=log_fn)
    except Exception as e:
        check_for_restart(e, _failed_before_bool, opt_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        run_relax_opt(atoms_obj, opt_path, opter_ase_fn, get_calc_fn,
                      fmax_float=fmax_float, max_steps_int=max_steps_int, log_fn=log_fn, _failed_before_bool=True)


def setup_img_dirs(neb_path, nImages, restart_bool=False, log_fn=log_def):
    img_dirs = []
    for j in range(nImages):
        img_dir_str = opj(neb_path, str(j))
        img_dirs.append(img_dir_str)
        if restart_bool:
            if not (ope(opj(img_dir_str, "POSCAR"))) and (ope(opj(img_dir_str, "CONTCAR"))):
                log_fn(f"Restart NEB requested but dir for image {j} does not appear to have a structure to use")
                log_fn(f"(Image {j}'s directory is {img_dir_str}")
                log_fn(f"Ignoring restart NEB request")
                restart_bool = False
        if not restart_bool:
            if ope(img_dir_str):
                log_fn(f"Restart NEB set to false but dir for image {j} found - resetting {img_dir_str}")
                remove_dir_recursive(img_dir_str, log_fn=log_fn)
            os.mkdir(img_dir_str)
    return img_dirs, restart_bool

def get_img_atoms(img_path_list, i, img_dir, pbc_bool_list, restart_bool=False, log_fn=log_def):
    failed = False
    try:
        img = get_atoms(img_dir, pbc_bool_list, restart_bool=restart_bool, log_fn=log_fn)
    except Exception as e:
        print(e)
        img = get_atoms(img_path_list[i-1], pbc_bool_list, restart_bool=restart_bool, log_fn=log_fn)
        write(opj(img_dir, "POSCAR"), img, format="vasp")
        failed = True
    return img, failed


def setup_neb_imgs(img_path_list, pbc_bool_list, get_calc_fn, log_fn=log_def, restart_bool=False):
    imgs = []
    interpolate = False
    for i in range(len(img_path_list)):
        img_dir = img_path_list[i]
        log_fn(f"Looking for structure for image {i} in {img_dir}")
        img, failed = get_img_atoms(img_path_list, i, img_dir, pbc_bool_list, restart_bool=False, log_fn=log_def)
        if failed:
            interpolate = True
        img.set_calculator(get_calc_fn(img_path_list[i]))
        imgs.append(img)
    return imgs, interpolate

def writing_bounding_images(start_struc, end_struc, images, neb_path):
    start_atoms = read(start_struc, format="vasp")
    end_atoms = read(end_struc, format="vasp")
    start_dir = opj(neb_path, str(0))
    end_dir = opj(neb_path, str(images-1))
    write(opj(start_dir, "POSCAR"), start_atoms, format="vasp")
    write(opj(end_dir, "POSCAR"), end_atoms, format="vasp")





def setup_neb(start_struc, end_struc, nImages, pbc, get_calc_fn, neb_path, k_float, neb_method_str, inter_method_str, gpu,
              opter_ase_fn=FIRE, restart_bool=False, use_ci_bool=False, log_fn=log_def,
              freeze_base=False, freeze_tol=0, freeze_count=0):
    if restart_bool:
        if not ope(opj(neb_path,"hessian.pckl")):
            log_fn(f"Restart NEB requested but no hessian pckl found - ignoring restart request")
            restart_bool = False
    log_fn(f"Setting up image directories in {neb_path}")
    check_submit(gpu, getcwd(), "neb", log_fn=log_fn)
    img_dirs, restart_bool = setup_img_dirs(neb_path, nImages, restart_bool=restart_bool, log_fn=log_fn)
    log_fn("Writing bounding images")
    writing_bounding_images(start_struc, end_struc, nImages, neb_path)
    log_fn(f"Creating image objects")
    imgs_atoms_list, interpolate = setup_neb_imgs(img_dirs, pbc, get_calc_fn, restart_bool=restart_bool, log_fn=log_fn)
    neb = NEB(imgs_atoms_list, parallel=False, climb=use_ci_bool, k=k_float, method=neb_method_str)
    if interpolate:
        neb.interpolate(apply_constraint=True, method=inter_method_str)
        for i in range(nImages):
            write(opj(img_dirs[i], "POSCAR"), neb.images[i], format="vasp")
    if freeze_base:
        for i in range(nImages):
            add_freeze_surf_base_constraint(neb.images[i], freeze_base=freeze_base, ztol=freeze_tol, freeze_count=freeze_count, log_fn=log_fn)
    log_fn(f"Creating optimizer object")
    dyn = neb_optimizer(neb, neb_path, opter=opter_ase_fn)
    log_fn(f"Attaching log functions to optimizer object")
    traj = Trajectory(opj(neb_path, "neb.traj"), 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    log_fn(f"Attaching log functions to each image")
    for i, path in enumerate(img_dirs):
        dyn.attach(Trajectory(opj(path, f'opt-{i}.traj'), 'w', imgs_atoms_list[i],
                              properties=['energy', 'forces']))
        dyn.attach(lambda img, img_dir: _write_contcar(img, img_dir),
                   interval=1, img_dir=img_dirs[i], img=imgs_atoms_list[i])
        dyn.attach(lambda img, img_dir: _write_img_opt_iolog(img, img_dir, log_fn),
                   interval=1, img_dir=img_dirs[i], img=imgs_atoms_list[i])
    return dyn, restart_bool



def main():
    nid = read_neb_inputs()
    restart = nid["restart"]
    gpu = nid["gpu"]
    pseudoSet = nid["pseudoSet"]
    pbc = nid["pbc"]
    nImages = nid["images"]
    use_ci = nid["ci"]
    k = nid["k"]
    neb_method = nid["neb_method"]
    interp_method = nid["interp_method"]
    max_steps = nid["max_steps"]
    fmax = nid["fmax"]
    bias = nid["bias"]
    work_dir = nid["work_dir"]
    freeze_base = nid["freeze_base"]
    freeze_tol = nid["freeze_tol"]
    freeze_count = nid["freeze_count"]
    chdir(work_dir)
    neb_log = get_log_fn(work_dir, "neb", False, restart=restart)
    start_struc = _check_structure(nid["start_struc"], work_dir, log_fn=neb_log)
    end_struc = _check_structure(nid["end_struc"], work_dir, log_fn=neb_log)
    ####################################################################################################################
    neb_log(f"Reading JDFTx commands")
    cmds = get_cmds_dict(work_dir, ref_struct=start_struc, log_fn=neb_log, pbc=pbc, bias=bias)
    cmds = cmds_dict_to_list(cmds)
    exe_cmd = get_exe_cmd(gpu, neb_log)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, pseudoSet=pseudoSet, debug=False, log_fn=neb_log)
    ####################################################################################################################
    neb_log("Beginning NEB setup")
    neb_dir = opj(work_dir, "neb")
    if not ope(neb_dir):
        neb_log("No NEB dir found - setting restart to False for NEB")
        restart = False
        mkdir(neb_dir)
    dyn_neb, skip_to_neb = setup_neb(start_struc, end_struc, nImages, pbc, get_calc, neb_dir, k, neb_method, interp_method, gpu,
                                     opter_ase_fn=FIRE, restart_bool=restart, use_ci_bool=use_ci, log_fn=neb_log)
    neb_log("Running NEB now")
    dyn_neb.run(fmax=fmax, steps=max_steps)
    neb_log(f"finished neb in {dyn_neb.nsteps}/{max_steps} steps")




if __name__ == '__main__':
    if debug:
        work_dir = "E:\\volD\\scratch_backup\\perl\\deepdive\\GBRV\\calcs\\nebs\\debug"
        chdir(work_dir)
    main()


