import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from os.path import join as opj
from os.path import exists as ope
import numpy as np
import shutil
from ase.neb import NEB
import time
from scripts.out_to_logx import get_do_cell
from helpers.generic_helpers import get_int_dirs, copy_state_files, atom_str, get_cmds, get_int_dirs_indices
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, _write_contcar, add_bond_constraints, optimizer
from helpers.generic_helpers import dump_template_input, _get_calc, get_exe_cmd, get_log_fn, copy_file, log_def
from helpers.generic_helpers import _write_logx, _write_opt_log, check_for_restart, finished_logx, sp_logx, bond_str
from helpers.generic_helpers import remove_dir_recursive, get_ionic_opt_cmds
from helpers.se_neb_helpers import get_fs, has_max, check_poscar, neb_optimizer, fix_step_size

se_neb_template = ["k: 0.1 # Spring constant for band forces in NEB step",
                   "neb method: spline # idk, something about how forces are projected out / imposed",
                   "scan: 1, 5, 10, 0.23 # (1st atom index (counting from 1 (1-based indexing)), 2nd atom index, number of steps, step size)",
                   "# target: 1.0 # (Not implemented yet) Modifies step size such that the final step's bond length matches",
                   "the target length",
                   "guess type: 0 # how structures are generated for the start of each bond scan step",
                   "# 0 = only move first atom (starting from previous step optimized geometry)",
                   "# 1 = only move second atom (starting from previous step optimized geometry)",
                   "# 2 = move both atoms equidistantly (starting from previous step optimized geometry)",
                   "# 3 = move all atoms following trajectory of previous two steps' optimized geometry (bond length enforced by a type 2 guess)"
                   "restart: 3 # step number to resume (if not given, this will be found automatically)",
                   "# restart: neb # would trigger a restart for the neb if scan has been completed"
                   "max_steps: 100 # max number of steps for scan opts",
                   "neb max steps: 30 # max number of steps for neb opt",
                   "fmax: 0.05 # fmax perameter for both neb and scan opt",
                   "pbc: True, true, false # which lattice vectors to impose periodic boundary conditions on",
                   "relax: start, end # start optimizes given structure without frozen bond before scanning bond, end ",
                   "optimizes final structure without frozen bond",
                   "# safe mode: True # (Not implemented yet) If end is relaxed and bond length becomes",]

def read_se_neb_inputs(fname="se_neb_inputs"):
    if not ope(fname):
        dump_template_input(fname, se_neb_template, os.getcwd())
        raise ValueError(f"No se neb input supplied: dumping template {fname}")
    k = 1.0
    neb_method = "spline"
    interp_method = "linear"
    lookline = None
    restart_at = None
    restart_neb = False
    max_steps = 100
    neb_max_steps = None
    fmax = 0.01
    work_dir = None
    follow = False
    debug = False
    relax_start = True
    relax_end = True
    inputs = get_inputs_list(fname)
    pbc = [True, True, False]
    guess_type = 2
    target = None
    safe_mode = False
    for input in inputs:
        key, val = input[0], input[1]
        if "scan" in key:
            lookline = val.split(",")
        if "restart" in key:
            if "neb" in val:
                restart_neb = True
            else:
                restart_at = int(val.strip())
        if "debug" in key:
            restart_bool_str = val
            debug = "true" in restart_bool_str.lower()
        if "work" in key:
            work_dir = val.strip()
        if ("method" in key) and ("neb" in key):
            neb_method = val.strip()
        if ("method" in key) and ("interp" in key):
            interp_method = val.strip()
        if key.lower()[0] == "k":
            k = float(val.strip())
        if "fix" in key:
            lsplit = val.split(",")
            fix_pairs = []
            for atom in lsplit:
                try:
                    fix_pairs.append(int(atom))
                except ValueError:
                    pass
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
            if guess_type == 3:
                follow = True
                guess_type = 2
    atom_pair = [int(lookline[0]) - 1, int(lookline[1]) - 1] # Convert to 0-based indexing
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    if restart_neb:
        restart_at = scan_steps + 1
    if neb_max_steps is None:
        neb_max_steps = int(max_steps / 10.)
    work_dir = fix_work_dir(work_dir)
    return atom_pair, scan_steps, step_length, restart_at, work_dir, follow, debug, max_steps, fmax, neb_method,\
        interp_method, k, neb_max_steps, pbc, relax_start, relax_end, guess_type, target, safe_mode


def get_atoms_prep_follow(atoms, prev_2_out, atom_pair, target_length):
    atoms_prev = read(prev_2_out, format="vasp")
    dir_vecs = []
    for i in range(len(atoms.positions)):
        dir_vecs.append(atoms.positions[i] - atoms_prev.positions[i])
    for i in range(len(dir_vecs)):
        atoms.positions[i] += dir_vecs[i]
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    cur_length = np.linalg.norm(dir_vec)
    should_be_0 = target_length - cur_length
    if not np.isclose(should_be_0, 0.0):
        atoms.positions[atom_pair[1]] += dir_vec * (should_be_0) / np.linalg.norm(dir_vec)
    return atoms


def _prep_input(step_idx, atom_pair, step_length, start_length, follow, step_dir, scan_dir, log_func=log_def, guess_type=1):
    step_prev_1_dir = opj(scan_dir, str(step_idx-1))
    step_prev_2_dir = opj(scan_dir, str(step_idx - 2))
    prev_1_out = opj(step_prev_1_dir, "CONTCAR")
    prev_2_out = opj(step_prev_2_dir, "CONTCAR")
    print_str = f"Prepared structure for step {step_idx} with"
    target_length = start_length + (step_idx*step_length)
    atoms = read(prev_1_out, format="vasp")
    if step_idx <= 1:
        follow = False
    if follow:
        print_str += " atom momentum followed"
        atoms = get_atoms_prep_follow(atoms, prev_2_out, atom_pair, target_length)
    else:
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        dir_vec *= step_length / np.linalg.norm(dir_vec)
        if guess_type == 0:
            print_str += f" only {atom_str(atoms, atom_pair[0])} moved"
            atoms.positions[atom_pair[1]] += dir_vec
        elif guess_type == 1:
            print_str += f" only {atom_str(atoms, atom_pair[0])} moved"
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
        elif guess_type == 2:
            print_str += f" only {atom_str(atoms, atom_pair[0])} and {atom_str(atoms, atom_pair[1])} moved equidistantly"
            dir_vec *= 0.5
            atoms.positions[atom_pair[1]] += dir_vec
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
    write(opj(step_dir, "POSCAR"), atoms, format="vasp")
    log_func(print_str)



def get_start_dist(scan_dir, atom_pair, restart=False, log_fn=log_def):
    dir0 = opj(scan_dir, "0")
    atoms = get_atoms(dir0, [False,False,False], restart_bool=restart, log_fn=log_fn)
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    start_dist = np.linalg.norm(dir_vec)
    log_fn(f"Bond {bond_str(atoms, atom_pair[0], atom_pair[1])} starting at {start_dist}")
    return start_dist


def do_relax_start(relax_start_bool, scan_dir_path, get_calc_fn, log_fn=log_def, fmax_float=0.05, max_steps_int=100):
    if relax_start_bool:
        dir0 = opj(scan_dir_path, "0")
        se_log(f"Relaxing initial geometry in {dir0}")
        atoms = get_atoms(dir0, pbc, restart_bool=True, log_fn=log_fn)
        run_relax_opt(atoms, dir0, FIRE, get_calc_fn, fmax_float=fmax_float, max_steps_int=max_steps_int, log_fn=log_fn)


def do_relax_end(scan_steps_int, scan_dir_str, restart_idx, pbc_list_of_bool, get_calc_fn, log_fn=log_def,
                 fmax_float=0.05, max_steps_int=100):
    end_idx = scan_steps_int
    end_dir = opj(scan_dir_str, str(end_idx))
    prev_dir = opj(scan_dir_str, str(end_idx - 1))
    if (not ope(opj(end_dir, "POSCAR"))) or (not os.path.isdir(end_dir)):
        os.mkdir(end_dir)
        restart_end = False
        copy_state_files(prev_dir, end_dir, log_fn=log_fn)
        prep_input(scan_steps_int, end_dir)
    else:
        restart_end = (end_idx == restart_idx) and (not is_done(end_dir, end_idx))
    atoms = get_atoms(end_dir, pbc_list_of_bool, restart_bool=restart_end, log_fn=log_fn)
    run_relax_opt(atoms, end_dir, FIRE, get_calc_fn, fmax_float=fmax_float, max_steps_int=max_steps_int, log_fn=log_fn)



def finished(dir_path):
    with open(opj(dir_path, f"finished_{os.path.basename(dir_path)}.txt"), "w") as f:
        f.write("done")


def is_done(dir_path, idx):
    return ope(opj(dir_path, f"finished_{idx}.txt"))


def get_restart_idx(restart_idx, scan_dir):
    if not restart_idx is None:
        return restart_idx
    else:
        restart_idx = 0
        if not ope(scan_dir):
            return restart_idx
        else:
            int_dirs = get_int_dirs(scan_dir)
            int_dirs_indices = get_int_dirs_indices(int_dirs)
            for i in range(len(int_dirs)):
                look_dir = int_dirs[int_dirs_indices[i]]
                if is_done(look_dir, i):
                    restart_idx = i
                else:
                    return restart_idx


def get_atoms(dir_path, pbc_list_of_bool, restart_bool=False, log_fn=log_def):
    _abort = False
    POSCAR = opj(dir_path, "POSCAR")
    CONTCAR = opj(dir_path, "CONTCAR")
    if restart_bool:
        if ope(CONTCAR):
            atoms_obj = read(CONTCAR, format="vasp")
            log_fn(f"Found CONTCAR in {dir_path}")
        elif ope(POSCAR):
            atoms_obj = read(POSCAR, format="vasp")
            log_fn(f"Could not find CONTCAR in {dir_path} - using POSCAR instead")
        else:
            _abort = True
    else:
        if ope(POSCAR):
            atoms_obj = read(POSCAR, format="vasp")
            log_fn(f"Found CONTCAR in {dir_path}")
        elif ope(CONTCAR):
            atoms_obj = read(CONTCAR, format="vasp")
            log_fn(f"Could not find start POSCAR in {dir_path} - using found CONTCAR instead")
        else:
            _abort = True
    if _abort:
        log_fn(f"Could not find structure from {dir_path} - aborting")
        assert False
    atoms_obj.pbc = pbc_list_of_bool
    log_fn(f"Setting pbc for atoms to {pbc_list_of_bool}")
    return atoms_obj


def run_opt_runner(atoms_obj, root_path, opter, log_fn = log_def, fmax=0.05, max_steps=100):
    dyn = optimizer(atoms_obj, root_path, opter)
    traj = Trajectory(opj(root_path, "opt.traj"), 'w', atoms_obj, properties=['energy', 'forces', 'charges'])
    logx = opj(root_path, "opt.logx")
    do_cell = get_do_cell(atoms_obj.pbc)
    dyn.attach(traj.write, interval=1)
    dyn.attach(lambda: _write_contcar(atoms_obj, root_path), interval=1)
    dyn.attach(lambda: _write_logx(atoms_obj, logx, dyn, max_steps, do_cell=do_cell), interval=1)
    dyn.attach(lambda: _write_opt_log(atoms_obj, dyn, max_steps, log_fn), interval=1)
    log_fn("Optimization starting")
    log_fn(f"Fmax: {fmax}, max_steps: {max_steps}")
    dyn.run(fmax=fmax, steps=max_steps)
    log_fn(f"Finished in {dyn.nsteps}/{max_steps}")
    finished_logx(atoms_obj, logx, dyn.nsteps, max_steps)
    sp_logx(atoms_obj, "sp.logx", do_cell=do_cell)
    finished(root_path)


def run_relax_opt(atoms_obj, opt_dir_path, opter_ase_fn, get_calc_fn,
                  fmax_float=0.05, max_steps_int=100, log_fn=log_def, _failed_before_bool=False):
    atoms_obj.set_calculator(get_calc_fn(opt_dir_path))
    run_again = False
    try:
        run_opt_runner(atoms_obj, opt_dir_path, opter_ase_fn, fmax=fmax_float, max_steps=max_steps_int, log_fn=log_fn)
    except Exception as e:
        assert check_for_restart(e, _failed_before_bool, opt_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        run_relax_opt(atoms_obj, opt_dir_path, opter_ase_fn, get_calc_fn,
                      fmax_float=fmax_float, max_steps_int=max_steps_int, log_fn=log_fn, _failed_before_bool=True)


def run_step(atoms_obj, step_dir_path, fix_pair_list_of_int, get_calc_fn, opter_ase_fn,
             fmax_float=0.1, max_steps_int=50, log_fn=log_def, _failed_before_bool=False):
    run_again = False
    add_bond_constraints(atoms_obj, fix_pair_list_of_int, log_fn=log_fn)
    atoms_obj.set_calculator(get_calc_fn(step_dir_path))
    try:
        run_opt_runner(atoms_obj, step_dir_path, opter_ase_fn, log_fn=log_fn, fmax=fmax_float, max_steps=max_steps_int)
    except Exception as e:
        log_fn(e)
        assert check_for_restart(e, _failed_before_bool, step_dir_path, log_fn=log_fn)
        run_again = True
        pass
    if run_again:
        run_step(atoms_obj, step_dir_path, fix_pair_list_of_int, get_calc_fn, opter_ase_fn,
                 fmax_float=fmax_float, max_steps_int=max_steps_int, log_fn=log_fn, _failed_before_bool=True)


def setup_img_dirs(neb_dir_path, scan_dir_path, scan_steps_int, restart_bool=False, log_fn=log_def):
    img_dirs = []
    for j in range(scan_steps_int):
        step_dir_str = opj(scan_dir_path, str(j))
        img_dir_str = opj(neb_dir_path, str(j))
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


def setup_neb_imgs(scab_steps_int, img_dirs_list_of_path, pbc_list_of_bool, get_calc_fn, log_fn=log_def, restart_bool=False):
    imgs = []
    for i in range(scab_steps_int):
        img_dir = img_dirs_list_of_path[i]
        log_fn(f"Looking for structure for image {i} in {img_dir}")
        img = get_atoms(img_dir, pbc_list_of_bool, restart_bool=restart_bool, log_fn=log_fn)
        img.set_calculator(get_calc_fn(img_dirs_list_of_path[i]))
        imgs.append(img)
    return imgs


def setup_neb(scan_steps_int, k_float, neb_method_str, pbc_list_of_bool, get_calc_fn, neb_dir_path, scan_dir_path,
              opter_ase_fn=FIRE, restart_bool=False, use_ci_bool=False, log_fn=log_def):
    if restart_bool:
        if not ope(opj(neb_dir_path,"hessian.pckl")):
            log_fn(f"Restart NEB requested but no hessian pckl found - ignoring restart request")
            restart_bool = False
    log_fn(f"Setting up image directories in {neb_dir_path}")
    img_dirs, restart_bool = setup_img_dirs(neb_dir_path, scan_dir_path, scan_steps_int,
                                            restart_bool=restart_bool, log_fn=log_fn)
    log_fn(f"Creating image objects")
    imgs_list_of_atoms = setup_neb_imgs(scan_steps_int, img_dirs, pbc_list_of_bool, get_calc_fn,
                                        restart_bool=restart_bool, log_fn=log_fn)
    log_fn(f"Creating NEB object")
    neb = NEB(imgs_list_of_atoms, parallel=False, climb=use_ci_bool, k=k_float, method=neb_method_str)
    log_fn(f"Creating optimizer object")
    dyn = neb_optimizer(neb, neb_dir_path, opter=opter_ase_fn)
    log_fn(f"Attaching log functions to optimizer object")
    traj = Trajectory(opj(neb_dir_path, "neb.traj"), 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    log_fn(f"Attaching log functions to each image")
    for i in range(scan_steps_int):
        dyn.attach(Trajectory(opj(img_dirs[i], 'opt-' + str(i) + '.traj'), 'w', imgs_list_of_atoms[i],
                              properties=['energy', 'forces']))
        dyn.attach(lambda img, img_dir: _write_contcar(img, img_dir),
                   interval=1, img_dir=img_dirs[i], img=imgs_list_of_atoms[i])
    return dyn


def setup_scan_dir(work_dir_path, scan_dir_path, relax_start_bool, restart_at_idx, pbc_bool_list, log_fn=log_def):
    dir0 = opj(scan_dir_path, "0")
    if not ope(scan_dir_path):
        log_fn("Creating scan directory")
        os.mkdir(scan_dir_path)
    if not ope(dir0):
        log_fn(f"Setting up directory for step 0 (this is special for step 0 - please congratulate him)")
        os.mkdir(dir0)
        copy_state_files(work_dir_path, dir0)
        atoms_obj = get_atoms(work_dir_path, pbc_bool_list, restart_bool=True, log_fn=log_fn)
        write(opj(dir0, "POSCAR"), atoms_obj, format="vasp")
    elif is_done(dir0, 0) and (not restart_at_idx == 0):
        log_fn(f"Step 0 appears to be done and we're restarting this scan beyond step 0")
        relax_start_bool = False
    log_fn("Checking for scan steps to be overwritten")
    int_dirs = get_int_dirs(work_dir_path)
    for dir_path in int_dirs:
        idx = os.path.basename(dir_path)
        if idx > restart_at_idx:
            log_fn(f"Step {idx} comes after requested restart index {restart_at_idx}")
            remove_dir_recursive(dir_path)
    return relax_start_bool


if __name__ == '__main__':
    atom_pair, scan_steps, step_length, restart_at, work_dir, follow, debug, max_steps, fmax, neb_method,\
        interp_method, k, neb_max_steps, pbc, relax_start, relax_end, guess_type, target, safe_mode = read_se_neb_inputs()
    os.chdir(work_dir)
    scan_dir = opj(work_dir, "scan")
    restart_at = get_restart_idx(restart_at, scan_dir) # If was none, finds most recently converged step
    restart = restart_at > 0
    skip_to_neb = (restart_at > scan_steps)
    se_log = get_log_fn(work_dir, "se_neb", False, restart=restart)
    if skip_to_neb:
        se_log("Will restart at NEB")
    else:
        se_log(f"Will restart at {restart_at}")
    if restart_at == 0:
        se_log(f"Making sure atoms in input structure are ordered")
        check_poscar(work_dir, se_log)
    ####################################################################################################################
    se_log(f"Reading JDFTx commands")
    cmds = get_cmds(work_dir, ref_struct="POSCAR")
    if not debug:
        from JDFTx import JDFTx
        exe_cmd = get_exe_cmd(True, se_log)
        get_calc = lambda root: _get_calc(exe_cmd, cmds, root, JDFTx,
                                          debug=debug, log_fn=se_log)
    else:
        exe_cmd = " "
        from ase.calculators.emt import EMT as debug_calc
        get_calc = lambda root: _get_calc(exe_cmd, cmds, root, None,
                                          debug=debug, debug_fn=debug_calc, log_fn=se_log)
    ####################################################################################################################
    if not skip_to_neb:
        se_log("Entering scan")
        relax_start = setup_scan_dir(work_dir, scan_dir, relax_start, restart_at, pbc, log_fn=se_log)
        do_relax_start(relax_start, scan_dir, get_calc, log_fn=se_log, fmax_float=fmax, max_steps_int=max_steps)
        start_length = get_start_dist(scan_dir, atom_pair, log_fn=se_log, restart=relax_start)
        if not target is None:
            step_length = fix_step_size(start_length, target, scan_steps, log_fn=se_log)
        prep_input = lambda i, step_dir_var: _prep_input(i, atom_pair, step_length, start_length, follow, step_dir_var,
                                                         scan_dir, guess_type=guess_type, log_func=se_log)
        for i in list(range(scan_steps))[restart_at:]:
            if relax_start and (i == 0):
                continue
            step_dir = opj(scan_dir, str(i))
            restart_step = (i == restart_at) and (not is_done(step_dir, i))
            if (not ope(step_dir)) or (not os.path.isdir(step_dir)):
                os.mkdir(step_dir)
            if i > 0 and not restart_step:
                prev_step_dir = opj(scan_dir, str(i-1))
                copy_state_files(prev_step_dir, step_dir, log_fn=se_log)
                prep_input(i, step_dir)
            atoms = get_atoms(step_dir, pbc, restart_bool=restart_step, log_fn=se_log)
            run_step(atoms, step_dir, atom_pair, get_calc, FIRE,
                     fmax_float=fmax, max_steps_int=max_steps, log_fn=se_log)
        if relax_end:
            do_relax_end(scan_steps, scan_dir, restart_at, pbc, get_calc,
                         log_fn=se_log, fmax_float=fmax, max_steps_int=max_steps)
    ####################################################################################################################
    se_log("Beginning NEB setup")
    neb_dir = opj(work_dir, "neb")
    if not ope(neb_dir):
        se_log("No NEB dir found - setting restart to False for NEB")
        skip_to_neb = False
        os.mkdir(neb_dir)
    use_ci = has_max(get_fs(scan_dir)) # Use climbing image if PES have a local maximum
    if use_ci:
        se_log("Local maximum found within scan - using climbing image method in NEB")
    dyn_neb = setup_neb(scan_steps + relax_end, k, neb_method, pbc, get_calc, neb_dir, scan_dir,
                        restart_bool=skip_to_neb, use_ci_bool=use_ci, log_fn=se_log)
    se_log("Running NEB now")
    dyn_neb.run(fmax=fmax, steps=neb_max_steps)
    se_log(f"finished neb in {dyn_neb.nsteps}/{neb_max_steps} steps")
