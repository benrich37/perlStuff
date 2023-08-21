import os
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from os.path import join as opj, exists as ope, isdir,  basename
from os import mkdir, getcwd,  chdir
from ase.neb import NEB
from helpers.generic_helpers import get_int_dirs, copy_state_files, get_cmds, get_int_dirs_indices, \
    get_atoms_list_from_out, get_do_cell, get_atoms
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, _write_contcar, optimizer
from helpers.generic_helpers import dump_template_input, get_log_fn, copy_file, log_def
from helpers.calc_helpers import _get_calc, get_exe_cmd
from helpers.generic_helpers import _write_opt_iolog, check_for_restart, get_bond_str, get_nrg
from helpers.generic_helpers import remove_dir_recursive, get_ionic_opt_cmds, check_submit
from helpers.geom_helpers import get_bond_length, get_property
from helpers.generic_helpers import death_by_nan, reset_atoms_death_by_nan
from helpers.logx_helpers import write_scan_logx, out_to_logx, _write_logx, finished_logx, sp_logx
from helpers.generic_helpers import add_freeze_list_constraints, copy_best_state_files, get_atom_str
from helpers.se_neb_helpers import get_fs, has_max, check_poscar, neb_optimizer, count_scan_steps, _prep_input, setup_scan_dir
from helpers.schedule_helpers import write_autofill_schedule, j_steps_key, freeze_list_key, read_schedule_file, \
    get_step_list, energy_key, properties_key, get_prop_idcs_list, append_results_as_comments, get_scan_steps_list_for_neb, get_neb_options

se_neb_template = ["k: 0.1 # Spring constant for band forces in NEB step",
                   "neb method: spline # idk, something about how forces are projected out / imposed",
                   "scan: 1, 5, 10, 0.23 # (1st atom index (counting from 1 (1-based indexing)), 2nd atom index, number of steps, step size)",
                   "# target: 1.0 # (Not implemented yet) Modifies step size such that the final step's bond length matches",
                   "# the target length",
                   "guess type: 0 # how structures are generated for the start of each bond scan step",
                   "# 0 = only move first atom (starting from previous step optimized geometry)",
                   "# 1 = only move second atom (starting from previous step optimized geometry)",
                   "# 2 = move both atoms equidistantly (starting from previous step optimized geometry)",
                   "# 3 = move all atoms following trajectory of previous two steps' optimized geometry (bond length enforced by a type 2 guess)"
                   "restart: 3 # step number to resume (if not given, this will be found automatically)",
                   "# restart: neb # would trigger a restart for the neb if scan has been completed"
                   "max_steps: 100 # max number of steps for scan opts",
                   "jdft steps: 5 # Number of ion-opt steps to take before running ASE opt",
                   "neb max steps: 30 # max number of steps for neb opt",
                   "fmax: 0.05 # fmax perameter for both neb and scan opt",
                   "pbc: True, true, false # which lattice vectors to impose periodic boundary conditions on",
                   "relax: start, end # start optimizes given structure without frozen bond before scanning bond, end ",
                   "# optimizes final structure without frozen bond",
                   "gpu: False"
                   "# safe mode: True # (Not implemented yet) If end is relaxed, scan images with bond lengths exceeding/smaller than this length",]

def read_se_neb_inputs(fname="se_neb_inputs"):
    if not ope(fname):
        dump_template_input(fname, se_neb_template, getcwd())
        raise ValueError(f"No se neb input supplied: dumping template {fname}")
    k = 1.0
    neb_method = "spline"
    lookline = None
    restart_at = None
    restart_neb = False
    max_steps = 100
    neb_max_steps = None
    fmax = 0.01
    work_dir = None
    relax_start = False
    relax_end = False
    inputs = get_inputs_list(fname)
    pbc = [True, True, False]
    guess_type = 2
    target = None
    safe_mode = False
    jdft_steps = 5
    schedule = False
    gpu = False
    for input in inputs:
        key, val = input[0], input[1]
        if "gpu" in key:
            gpu = "true" in val.lower()
        if "scan" in key:
            if "schedule" in val:
                schedule = True
            else:
                lookline = val.split(",")
        if "restart" in key:
            if "neb" in val:
                restart_neb = True
            else:
                restart_at = int(val.strip())
        if "work" in key:
            work_dir = val.strip()
        if ("method" in key) and ("neb" in key):
            neb_method = val.strip()
        if key.lower()[0] == "k":
            k = float(val.strip())
        if "max" in key:
            if "steps" in key:
                if "neb" in key:
                    neb_max_steps = int(val.strip())
                else:
                    max_steps = int(val.strip())
            elif ("force" in key) or ("fmax" in key):
                fmax = float(val.strip())
        if "pbc" in key:
            pbc = read_pbc_val(val)
        if "relax" in key:
            if "start" in val:
                relax_start = True
            if "end" in val:
                relax_end = True
        if ("guess" in key) and ("type" in key):
            guess_type = int(val)
        if ("jdft" in key) and ("step" in key):
            jdft_steps = int(val)
        if ("safe" in key) and ("mode" in key):
            safe_mode = "true" in val.lower()
    atom_idcs = None
    scan_steps = None
    step_length = None
    if not lookline is None:
        atom_idcs, scan_steps, step_length = parse_lookline(lookline)
    if neb_max_steps is None:
        neb_max_steps = int(max_steps / 10.)
    work_dir = fix_work_dir(work_dir)
    if schedule:
        scan_steps = count_scan_steps(work_dir)
    return atom_idcs, scan_steps, step_length, restart_at, restart_neb, work_dir, max_steps, fmax, neb_method,\
        k, neb_max_steps, pbc, relax_start, relax_end, guess_type, target, safe_mode, jdft_steps, schedule, gpu


def parse_lookline(lookline):
    step_length = float(lookline[-1])
    scan_steps = float(lookline[-2])
    atom_idcs = []
    for idx in lookline[:-2]:
        atom_idcs.append(int(idx - 1))
    return atom_idcs, scan_steps, step_length



def finished(dir_path):
    with open(opj(dir_path, f"finished_{basename(dir_path)}.txt"), "w") as f:
        f.write("done")


def is_done(dir_path, idx):
    return ope(opj(dir_path, f"finished_{idx}.txt"))


def get_restart_idx(restart_idx, scan_path, log_fn=log_def):
    if not restart_idx is None:
        log_fn(f"Restart index specified at {restart_idx}")
        return restart_idx
    else:
        restart_idx = 0
        if not ope(scan_path):
            return restart_idx
        else:
            int_dirs = get_int_dirs(scan_path)
            int_dirs_indices = get_int_dirs_indices(int_dirs)
            for i in range(len(int_dirs)):
                look_dir = int_dirs[int_dirs_indices[i]]
                if ope(look_dir):
                    if is_done(look_dir, i):
                        restart_idx = i
                    else:
                        return i
                else:
                    return i
            return restart_idx


def run_preopt(atoms_obj, root_path, log_fn=log_def):
    outfile = opj(root_path, "out")
    log_fn("JDFTx pre-optimization starting")
    atoms_obj.get_forces()
    log_fn("JDFTx pre-optimization finished")
    jdft_opt = opj(root_path, "pre_opt")
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
        atoms_obj = run_preopt(atoms_obj, step_path, log_fn=log_fn)
    atoms_obj.set_calculator(get_calc_fn(step_path))
    run_opt_runner(atoms_obj, step_path, opter, log_fn=log_fn, fmax=fmax, max_steps=max_steps)



def read_instructions_run_step(instructions):
    freeze_list = instructions[freeze_list_key]
    j_steps = instructions[j_steps_key]
    return freeze_list, j_steps



def run_step(atoms_obj, step_path, instructions, get_jdft_opt_calc_fn, get_calc_fn, opter_ase_fn,
             fmax_float=0.1, max_steps_int=50, log_fn=log_def, _failed_before_bool=False):
    freeze_list, j_steps = read_instructions_run_step(instructions)
    run_again = False
    add_freeze_list_constraints(atoms_obj, freeze_list, log_fn=log_fn)
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


def setup_img_dirs(neb_path, scan_path, schedule, restart_bool=False, log_fn=log_def):
    img_dirs = []
    scan_steps_list = get_scan_steps_list_for_neb(schedule)
    for j, scan_idx in enumerate(scan_steps_list):
        step_dir_str = opj(scan_path, str(scan_idx))
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
            copy_state_files(step_dir_str, img_dir_str, log_fn=log_fn)
            copy_file(opj(step_dir_str, "CONTCAR"), opj(img_dir_str, "POSCAR"), log_fn=log_fn)
    return img_dirs, restart_bool


def setup_neb_imgs(img_path_list, pbc_bool_list, get_calc_fn, log_fn=log_def, restart_bool=False):
    imgs = []
    for i in range(len(img_path_list)):
        img_dir = img_path_list[i]
        log_fn(f"Looking for structure for image {i} in {img_dir}")
        img = get_atoms(img_dir, pbc_bool_list, restart_bool=restart_bool, log_fn=log_fn)
        img.set_calculator(get_calc_fn(img_path_list[i]))
        imgs.append(img)
    return imgs



def setup_neb(schedule, pbc_bool_list, get_calc_fn, neb_path, scan_path,
              opter_ase_fn=FIRE, restart_bool=False, use_ci_bool=False, log_fn=log_def):
    if restart_bool:
        if not ope(opj(neb_path,"hessian.pckl")):
            log_fn(f"Restart NEB requested but no hessian pckl found - ignoring restart request")
            restart_bool = False
    log_fn(f"Setting up image directories in {neb_path}")
    img_dirs, restart_bool = setup_img_dirs(neb_path, scan_path, schedule,
                                            restart_bool=restart_bool, log_fn=log_fn)
    log_fn(f"Creating image objects")
    imgs_atoms_list = setup_neb_imgs(img_dirs, pbc_bool_list, get_calc_fn, restart_bool=restart_bool, log_fn=log_fn)
    log_fn(f"Creating NEB object")
    k_float, neb_method_str = get_neb_options(schedule)
    neb = NEB(imgs_atoms_list, parallel=False, climb=use_ci_bool, k=k_float, method=neb_method_str)
    log_fn(f"Creating optimizer object")
    dyn = neb_optimizer(neb, neb_path, opter=opter_ase_fn)
    log_fn(f"Attaching log functions to optimizer object")
    traj = Trajectory(opj(neb_path, "neb.traj"), 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    log_fn(f"Attaching log functions to each image")
    for i, path in enumerate(img_dirs):
        dyn.attach(Trajectory(opj(path, 'opt-' + str(i) + '.traj'), 'w', imgs_atoms_list[i],
                              properties=['energy', 'forces']))
        dyn.attach(lambda img, img_dir: _write_contcar(img, img_dir),
                   interval=1, img_dir=img_dirs[i], img=imgs_atoms_list[i])
    return dyn, restart_bool


def scan_is_finished(scan_dir, log_fn=log_def):
    int_dirs = get_int_dirs(scan_dir)
    idcs = get_int_dirs_indices(int_dirs)
    last_step_path = int_dirs[idcs[-1]]
    last_is_finished = is_done(last_step_path, idcs[-1])
    return last_is_finished


def get_properties_for_step(schedule, idx, step_dir):
    atoms = get_atoms(step_dir, [False, False, False], restart_bool=True, log_fn=lambda s: None)
    prop_idcs_list = get_prop_idcs_list(schedule, idx)
    tmp = []
    for idcs in prop_idcs_list:
        prop = get_property(atoms, idcs)
        prop_proper = []
        for idx in idcs:
            prop_proper.append(get_atom_str(atoms, idx))
        prop_proper.append(prop)
        tmp.append(prop_proper)
    return tmp


def update_results_to_schedule(schedule, scan_dir, work_dir, log_fn=log_def):
    step_dirs = get_int_dirs(scan_dir)
    log_fn(f"Adding current results as comments to schedule")
    for step_dir in step_dirs:
        idx = int(basename(step_dir))
        if is_done(step_dir, idx):
            schedule[str(idx)][energy_key] = get_nrg(step_dir)
            schedule[str(idx)][properties_key] = get_properties_for_step(schedule, idx, step_dir)
    append_results_as_comments(schedule, work_dir)



def main():
    atom_idcs, scan_steps, step_length, restart_at, restart_neb, work_dir, max_steps, fmax, neb_method, \
        k, neb_steps, pbc, relax_start, relax_end, guess_type, target, safe_mode, j_steps, schedule, gpu = read_se_neb_inputs()
    chdir(work_dir)
    if not schedule:
        write_autofill_schedule(atom_idcs, scan_steps, step_length, guess_type, j_steps, [atom_idcs], relax_start,
                                relax_end,
                                neb_steps, k, neb_method, work_dir)
    schedule = read_schedule_file(work_dir)
    scan_dir = opj(work_dir, "scan")
    restart_at = get_restart_idx(restart_at, scan_dir)  # If was none, finds most recently converged step
    restart = restart_at > 0
    skip_to_neb = (restart_at > scan_steps)
    se_log = get_log_fn(work_dir, "se_neb", False, restart=restart)
    update_results_to_schedule(schedule, scan_dir, work_dir, log_fn=se_log)
    if skip_to_neb:
        se_log("Will restart at NEB")
    else:
        se_log(f"Will restart at {restart_at}")
    if restart_at == 0:
        se_log(f"Making sure atoms in input structure are ordered")
        check_poscar(work_dir, se_log)
    step_list = get_step_list(schedule, restart_at)
    ####################################################################################################################
    se_log(f"Reading JDFTx commands")
    cmds = get_cmds(work_dir, ref_struct="POSCAR")
    exe_cmd = get_exe_cmd(gpu, se_log)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, debug=False, log_fn=se_log)
    get_ionopt_calc = lambda root, nMax: _get_calc(exe_cmd, get_ionic_opt_cmds(cmds, nMax), root, debug=False,
                                                   log_fn=se_log)
    ####################################################################################################################
    if not skip_to_neb:
        se_log("Entering scan")
        setup_scan_dir(work_dir, scan_dir, restart_at, pbc, log_fn=se_log)
        prep_input = lambda step, step_dir_var: _prep_input(step, schedule, step_dir_var, scan_dir, work_dir,
                                                            log_fn=se_log)
        for i, step in enumerate(step_list):
            step_dir = opj(scan_dir, str(step))
            se_log(f"Running step {step} in {step_dir}")
            restart_step = (i == 0) and (not is_done(step_dir, i))
            if (not ope(step_dir)) or (not isdir(step_dir)):
                mkdir(step_dir)
                restart_step = False
            if restart_step:
                se_log(f"Restarting step")
            if step > 0 and not restart_step:
                prev_step_dir = opj(scan_dir, str(step - 1))
            else:
                prev_step_dir = work_dir
            copy_best_state_files([prev_step_dir, step_dir], step_dir, log_fn=se_log)
            prep_input(step, step_dir)
            atoms = get_atoms(step_dir, pbc, restart_bool=restart_step, log_fn=se_log)
            check_submit(gpu, getcwd(), "se_neb", log_fn=se_log)
            run_step(atoms, step_dir, schedule[str(step)], get_ionopt_calc, get_calc, FIRE,
                     fmax_float=fmax, max_steps_int=max_steps, log_fn=se_log)
            write_scan_logx(scan_dir, log_fn=se_log)
            update_results_to_schedule(schedule, scan_dir, work_dir, log_fn=se_log)
    ####################################################################################################################
    se_log("Beginning NEB setup")
    neb_dir = opj(work_dir, "neb")
    if not ope(neb_dir):
        se_log("No NEB dir found - setting restart to False for NEB")
        skip_to_neb = False
        mkdir(neb_dir)
    use_ci = has_max(get_fs(scan_dir))  # Use climbing image if PES has a local maximum
    if use_ci:
        se_log("Local maximum found within scan - using climbing image method in NEB")
    dyn_neb, skip_to_neb = setup_neb(schedule, pbc, get_calc, neb_dir, scan_dir,
                                     restart_bool=skip_to_neb, use_ci_bool=use_ci, log_fn=se_log)
    if skip_to_neb:
        check_submit(gpu, getcwd(), "se_neb", log_fn=se_log)
    se_log("Running NEB now")
    dyn_neb.run(fmax=fmax, steps=neb_steps)
    se_log(f"finished neb in {dyn_neb.nsteps}/{neb_steps} steps")




if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

