import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE
from ase.calculators.emt import EMT as debug_calc
import numpy as np
import shutil
from ase.neb import NEB
import time
from generic_helpers import read_inputs, insert_el, get_int_dirs, copy_rel_files, remove_restart_files, atom_str
from neb_scan_helpers import read_neb_scan_inputs, log_total_elapsed, _neb_scan_log, check_poscar, neb_optimizer


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


def optimizer(atoms, work = "", opt_alpha=150, logfile='opt.log'):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    logfile = work + logfile
    restart = work + "hessian.pckl"
    dyn = FIRE(atoms, logfile=logfile, restart=restart, a=(opt_alpha / 70) * 0.1, maxstep=0.1)
    return dyn


def bond_constraint(atoms, indices):
    atoms.set_constraint(FixBondLength(indices[0], indices[1]))
    cur_length = np.linalg.norm(atoms.positions[indices[0]] - atoms.positions[indices[1]])
    print_str = f"Fixed bond {atom_str(atoms, indices[0])} -"
    print_str += f" {atom_str(atoms, indices[1])} fixed to {cur_length:.{4}g} A"
    scan_log(print_str)
    return atoms


def prep_input(step_idx, atom_pair, step_length, start_length, follow, step_type=1):
    print_str = f"Prepared structure for step {step_idx} with"
    target_length = start_length + (step_idx*step_length)
    print(target_length)
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
    scan_log(print_str)




def run_step(step_dir, fix_pair, exe_cmd, inputs_cmds, debug=False, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(step_dir, "POSCAR"), format="vasp")
    atoms.pbc = [True, True, False]
    bond_constraint(atoms, fix_pair)
    calculator = set_calc(exe_cmd, inputs_cmds, outfile=step_dir, debug=debug)
    atoms.set_calculator(calculator)
    dyn = optimizer(atoms, logfile=os.path.join(step_dir, "opt.log"))
    traj = Trajectory(step_dir +'opt.traj', 'w', atoms, properties=['energy', 'forces'])
    dyn.attach(traj.write, interval=1)
    def write_contcar(a=atoms):
        a.write(step_dir +'CONTCAR', format="vasp", direct=True)
        insert_el(step_dir +'CONTCAR')
    dyn.attach(write_contcar, interval=1)
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
        copy_rel_files("./", "./0")


def _setup_mini_neb_dirs(neb_dir, front, restart):
    if not restart:
        os.mkdir(neb_dir)
    img_dirs = []
    for i in range(front + 1):
        img_dir = neb_dir + str(i)
        img_dirs.append(img_dir)
        if not restart:
            os.mkdir(img_dir)
            copy_rel_files(f"./{str(i)}/", img_dir)
    return img_dirs

def _setup_mini_neb_imgs(front, img_dirs, exe_cmd, inputs_cmds, debug=False):
    imgs = []
    for i in range(front + 1):
        img = read(os.path.join(img_dirs[i], "CONTCAR"), format="vasp")
        img.pbc = [True, True, False]
        img.set_calculator(set_calc(exe_cmd, inputs_cmds, debug=debug, outfile=img_dirs[i]))
        imgs.append(img)
    return imgs

def setup_mini_neb(front, k, neb_method, exe_cmd, inputs_cmds, opter="FIRE", debug=False, restart = False):
    neb_dir = f"./iters/{str(front)}/"
    restart = restart and os.path.exists(neb_dir + "hessian.pckl")
    img_dirs = _setup_mini_neb_dirs(neb_dir, front, restart)
    imgs = _setup_mini_neb_imgs(front, img_dirs, exe_cmd, inputs_cmds, debug=debug)
    neb = NEB(imgs, parallel=False, climb=False, k=k, method=neb_method)
    dyn = neb_optimizer(neb, neb_dir, opt=opter)
    traj = Trajectory(neb_dir + 'neb.traj', 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    def write_contcar(img_dir, image):
        image.write(os.path.join(img_dir, 'CONTCAR'), format="vasp", direct=True)
        insert_el(os.path.join(img_dir, 'CONTCAR'))
    for i in range(front+1):
        dyn.attach(Trajectory(os.path.join(img_dirs[i], 'opt-' + str(i) + '.traj'), 'w', imgs[i], properties=['energy', 'forces']))
        dyn.attach(write_contcar, interval=1, img_dir=img_dirs[i], image=imgs[i])
    return dyn

def backup_mini_neb_to_cur(front):
    neb_dir = f"./iters/{str(front)}/"
    for i in range(front + 1):
        img_dir = os.path.join(neb_dir, str(i))
        copy_rel_files(img_dir, "./" + str(i))
def write_finished(front):
    with open(f"./{str(front)}/finished_{str(front)}.txt", "w") as f:
        f.write("done")

inputs_cmds = None
neb_time = 0
front_time = 0

if __name__ == '__main__':
    start1 = time.time()
    atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, \
        debug, max_steps, fmax, neb_method, interp_method, k, neb_max_steps = read_neb_scan_inputs()
    scan_log = lambda s: _neb_scan_log(s, work_dir, print_bool=debug)
    if not debug:
        from JDFTx import JDFTx
        jdftx_exe = os.environ['JDFTx_GPU']
        exe_cmd = 'srun ' + jdftx_exe
    else:
        exe_cmd = " "
    os.chdir(work_dir)
    check_poscar(work_dir)
    if os.path.exists("inputs"):
        inputs_cmds = read_inputs("inputs")
    prep_root(work_dir, restart_idx)
    start_length = get_start_dist(work_dir, atom_pair)
    scan_log(start_length)
    for front in list(range(scan_steps))[restart_idx:]:
        scan_log(f"################--------{str(front)}--------################")
        front_dir = f"./{str(front)}/"
        skip_to_neb = False
        if os.path.exists(front_dir + "finished_" + str(front) + ".txt"):
            scan_log("Frontier node" + str(front) + " already optimized - skipping to NEB")
            skip_to_neb = True
        else:
            if (not os.path.exists(front_dir)) or (not os.path.isdir(front_dir)):
                os.mkdir(front_dir)
            if front > 0:
                copy_rel_files(f"./{str(front - 1)}", f"./{str(front)}")
                prep_input(front, atom_pair, step_length, start_length, follow)
            start2 = time.time()
            try:
                run_step(work_dir + str(front) + "/", atom_pair, exe_cmd, inputs_cmds, debug=debug, fmax=fmax, max_steps=max_steps)
            except Exception as e:
                if front == 0:
                    scan_log(e)
                    scan_log("Trying once again without initial-state files present")
                    remove_restart_files("./0")
                    run_step(work_dir + str(front) + "/", atom_pair, exe_cmd, inputs_cmds, debug=debug, fmax=fmax,
                             max_steps=max_steps)
                    pass
                else:
                    scan_log(e)
                    assert False, str(e)
            write_finished(front)
            front_time += time.time() - start2
        if front >= 2:
            dyn = setup_mini_neb(front, k, neb_method, exe_cmd, inputs_cmds, debug=debug, restart = skip_to_neb)
            start2 = time.time()
            dyn.run(fmax=fmax, steps=neb_max_steps)
            scan_log(f"finished neb in {dyn.nsteps}/{neb_max_steps} steps")
            neb_time += time.time() - start2
            backup_mini_neb_to_cur(front)
        log_total_elapsed(start1, neb_time, front_time, work_dir)
