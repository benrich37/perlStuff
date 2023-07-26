import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import  FIRE
from os.path import join as opj
from os.path import exists as ope
import numpy as np
import shutil
from ase.neb import NEB
import time
from helpers.generic_helpers import get_int_dirs, copy_state_files, atom_str, get_cmds, get_int_dirs_indices
from helpers.generic_helpers import fix_work_dir, read_pbc_val, get_inputs_list, write_contcar, add_bond_constraints, optimizer
from helpers.generic_helpers import dump_template_input, _get_calc, get_exe_cmd
from helpers.neb_scan_helpers import _neb_scan_log, check_poscar, neb_optimizer
from helpers.se_neb_helpers import get_fs, has_max

se_neb_template = ["k: 0.1",
                   "neb method: spline",
                   "interp method: linear",
                   "scan: 3, 5, 10, 0.23",
                   "restart: 3",
                   "max_steps: 100",
                   "neb max steps: 30",
                   "fmax: 0.05",
                   "follow: False",
                   "pbc: True, true, false"]

def read_se_neb_inputs(fname="se_neb_inputs"):
    """
    nImages: 10
    restart: True
    initial: POSCAR_start
    final: POSCAR_end
    work: /pscratch/sd/b/beri9208/1nPt1H_NEB/calcs/surfs/H2_H2O_start/No_bias/scan_bond_test/
    k: 0.2
    neb_method: spline
    interp_method: linear
    fix_pair: 0, 5
    fmax: 0.03
    """
    if not ope(fname):
        dump_template_input(fname, se_neb_template, os.getcwd())
        raise ValueError(f"No se neb input supplied: dumping template {fname}")
    k = 1.0
    neb_method = "spline"
    interp_method = "linear"
    lookline = None
    restart_idx = None
    max_steps = 100
    neb_max_steps = None
    fmax = 0.01
    work_dir = None
    follow = False
    debug = False
    inputs = get_inputs_list(fname)
    pbc = [True, True, False]
    for input in inputs:
        key, val = input[0], input[1]
        if "scan" in key:
            lookline = val.split(",")
        if "restart" in key:
            restart_idx = int(val.strip())
        if "debug" in key:
            restart_bool_str = val
            debug = "true" in restart_bool_str.lower()
        if "work" in key:
            work_dir = val.strip()
        if ("method" in key) and ("neb" in key):
            neb_method = val.strip()
        if ("method" in key) and ("interp" in key):
            interp_method = val.strip()
        if "follow" in key:
            follow = "true" in val
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
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    if neb_max_steps is None:
        neb_max_steps = int(max_steps / 10.)
    work_dir = fix_work_dir(work_dir)
    return atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, debug, max_steps, fmax, neb_method, interp_method, k, neb_max_steps, pbc


def set_calc(exe_cmd, cmds, outfile=os.getcwd(), debug=False):
    if debug:
        return debug_calc()
    else:
        return JDFTx(
            executable=exe_cmd,
            pseudoSet="GBRV_v1.5",
            commands=cmds,
            outfile=outfile,
            ionic_steps=False,
            ignoreStress=True,
    )

def _prep_input(step_idx, atom_pair, step_length, start_length, follow, log_func, step_type):
    print_str = f"Prepared structure for step {step_idx} with"
    target_length = start_length + (step_idx*step_length)
    atoms = read(str(step_idx - 1) + "/CONTCAR", format="vasp")
    if step_idx <= 1:
        follow = False
    if follow:
        print_str += " atom momentum followed"
        atoms_prev = read(str(step_idx - 2) + "/CONTCAR", format="vasp")
        dir_vecs = []
        for i in range(len(atoms.positions)):
            dir_vecs.append(atoms.positions[i] - atoms_prev.positions[i])
        for i in range(len(dir_vecs)):
            atoms.positions[i] += dir_vecs[i]
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        cur_length = np.linalg.norm(dir_vec)
        should_be_0 = target_length - cur_length
        if not np.isclose(should_be_0, 0.0):
            atoms.positions[atom_pair[1]] += dir_vec*(should_be_0)/np.linalg.norm(dir_vec)
    else:
        dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
        dir_vec *= step_length / np.linalg.norm(dir_vec)
        if step_type == 0:
            print_str += f" only {atom_str(atoms, atom_pair[1])} moved"
            atoms.positions[atom_pair[1]] += dir_vec
        elif step_type == 1:
            print_str += f" only {atom_str(atoms, atom_pair[0])} and {atom_str(atoms, atom_pair[1])} moved equidistantly"
            dir_vec *= 0.5
            atoms.positions[atom_pair[1]] += dir_vec
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
        elif step_type == 2:
            print_str += f" only {atom_str(atoms, atom_pair[0])} moved"
            atoms.positions[atom_pair[0]] += (-1) * dir_vec
    write(str(step_idx) + "/POSCAR", atoms, format="vasp")
    log_func(print_str)




def run_step(step_dir, fix_pair, pbc, log_fn, debug=False, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(step_dir, "POSCAR"), format="vasp")
    atoms.pbc = pbc
    add_bond_constraints(atoms, fix_pair, log_fn=log_fn)
    atoms.set_calculator(get_calc(step_dir))
    dyn = optimizer(atoms, step_dir, FIRE)
    traj = Trajectory(opj(step_dir,'opt.traj'), 'w', atoms, properties=['energy', 'forces'])
    dyn.attach(traj.write, interval=1)
    dyn.attach(lambda a: write_contcar(a, step_dir), interval=1)
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        scan_log(f"finished frontier node in {str(dyn.nsteps)}/{max_steps} steps")
    except Exception as e:
        scan_log("couldnt run??")
        scan_log(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)

def get_start_dist(work_dir, atom_pair):
    atoms = read(work_dir + "0/POSCAR")
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    return np.linalg.norm(dir_vec)


def prep_root(work_dir, restart_idx):
    if not os.path.exists("iters"):
        scan_log("setting up iter directory")
        os.mkdir("./iters")
    elif restart_idx == 0:
        scan_log("resetting iter directory")
        shutil.rmtree("./iters")
        os.mkdir("./iters")
    else:
        int_dirs = get_int_dirs("./iters/")
        for dirr in int_dirs:
            if int(dirr.split('/')[-1]) >= restart_idx:
                scan_log("removing " + str(dirr))
                shutil.rmtree(dirr)
    int_dirs = get_int_dirs("./")
    for dirr in int_dirs:
        if int(dirr.split('/')[-1]) > restart_idx:
            scan_log("removing " + str(dirr))
            shutil.rmtree(dirr)
    if (not os.path.exists("./0")) or (not os.path.isdir("./0")):
        os.mkdir("./0")
    if restart_idx == 0:
        copy_state_files("./", "./0")


def _setup_mini_neb_dirs(neb_dir, front, restart):
    if not restart:
        os.mkdir(neb_dir)
    img_dirs = []
    for i in range(front + 1):
        img_dir = neb_dir + str(i)
        img_dirs.append(img_dir)
        if not restart:
            os.mkdir(img_dir)
            copy_state_files(f"./{str(i)}/", img_dir)
    return img_dirs

def _setup_mini_neb_imgs(front, img_dirs, exe_cmd, inputs_cmds, pbc, debug=False):
    imgs = []
    for i in range(front + 1):
        img = read(os.path.join(img_dirs[i], "CONTCAR"), format="vasp")
        img.pbc = pbc
        img.set_calculator(set_calc(exe_cmd, inputs_cmds, debug=debug, outfile=img_dirs[i]))
        imgs.append(img)
    return imgs

def setup_neb(front, k, neb_method, exe_cmd, inputs_cmds, pbc, opter="FIRE", debug=False, restart = False, use_ci = False):
    neb_dir = f"./neb/"
    restart = restart and os.path.exists(neb_dir + "hessian.pckl")
    img_dirs = _setup_mini_neb_dirs(neb_dir, front, restart)
    imgs = _setup_mini_neb_imgs(front, img_dirs, exe_cmd, inputs_cmds, pbc, debug=debug)
    neb = NEB(imgs, parallel=False, climb=use_ci, k=k, method=neb_method)
    dyn = neb_optimizer(neb, neb_dir, opt=opter)
    traj = Trajectory(neb_dir + 'neb.traj', 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    for i in range(front+1):
        dyn.attach(Trajectory(opj(img_dirs[i], 'opt-' + str(i) + '.traj'), 'w', imgs[i], properties=['energy', 'forces']))
        dyn.attach(lambda img_dir, image: write_contcar(image, img_dir), interval=1, img_dir=img_dirs[i], image=imgs[i])
    return dyn
def write_finished(front):
    with open(f"./{str(front)}/finished_{str(front)}.txt", "w") as f:
        f.write("done")

def is_done(dir, idx):
    return os.path.exists(os.path.join(dir, f"finished_{idx}.txt"))

def auto_restart_idx(restart_idx, work_dir):
    if not restart_idx is None:
        return restart_idx
    else:
        restart_idx = 0
        int_dirs = get_int_dirs(work_dir)
        int_dirs_indices = get_int_dirs_indices(int_dirs)
        for i in range(len(int_dirs)):
            look_dir = int_dirs[int_dirs_indices[i]]
            if is_done(look_dir, i):
                restart_idx = i
            else:
                return restart_idx

inputs_cmds = None
neb_time = 0
front_time = 0

if __name__ == '__main__':
    start1 = time.time()
    atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, \
        debug, max_steps, fmax, neb_method, interp_method, k, neb_max_steps, pbc = read_se_neb_inputs()
    restart_idx = auto_restart_idx(restart_idx, work_dir)
    scan_log = lambda s: _neb_scan_log(s, work_dir, print_bool=debug)
    cmds = get_cmds(work_dir)
    if not debug:
        from JDFTx import JDFTx
        exe_cmd = get_exe_cmd(True, scan_log)
        get_calc = lambda root: _get_calc(exe_cmd, cmds, root, JDFTx, debug=debug, log_fn=scan_log)
    else:
        exe_cmd = " "
        from ase.calculators.emt import EMT as debug_calc
        get_calc = lambda root: _get_calc(exe_cmd, cmds, root, None, debug=debug, debug_fn=debug_calc,
                                                log_fn=scan_log)
    os.chdir(work_dir)
    skip_to_neb = restart_idx == scan_steps
    if not skip_to_neb:
        start_length = get_start_dist(work_dir, atom_pair)
        prep_input = lambda i, atom_pair, step_length: _prep_input(i, atom_pair, step_length, start_length, follow, scan_log, 1)
        check_poscar(work_dir, scan_log)
        prep_root(work_dir, restart_idx)
        scan_log(start_length)
        for i in list(range(scan_steps))[restart_idx:]:
            if (not os.path.exists(f"./{str(i)}")) or (not os.path.isdir(f"./{str(i)}")):
                os.mkdir(f"./{str(i)}")
            if i > 0:
                copy_state_files(f"./{str(i - 1)}", f"./{str(i)}")
            if (i > 1):
                prep_input(i, atom_pair, step_length)
            run_step(work_dir + str(i) + "/", atom_pair, pbc, scan_log, fmax=fmax, max_steps=max_steps)
    if not os.path.exists("neb"):
        skip_to_neb = False
        os.mkdir("neb")
        for i in range(scan_steps):
            os.mkdir(f"neb/{i}")
            copy_state_files(str(i), f"neb/{i}")
    use_ci = has_max(get_fs(work_dir))
    dyn = setup_neb(scan_steps, k, neb_method, exe_cmd, inputs_cmds, pbc, debug=debug, restart=skip_to_neb, use_ci = use_ci)
    dyn.run(fmax=fmax, steps=neb_max_steps)
    scan_log(f"finished neb in {dyn.nsteps}/{neb_max_steps} steps")
