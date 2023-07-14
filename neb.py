import os
from ase.io import read, write
import subprocess
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE
from ase.calculators.emt import EMT as debug_calc
from ase.neb import NEB
from ase.build.tools import sort
import numpy as np
from datetime import datetime
import shutil
from generic_helpers import read_inputs, read_line_generic, dup_cmds, insert_el, get_int_dirs, log_generic, get_int_dirs_indices

def read_neb_inputs():
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
    nImages = None
    restart_bool = False
    work_dir = None
    initial = "initial"
    final = "final"
    k = 0.1
    neb_method = "spline"
    interp_method = "linear"
    fix_pairs = None
    fmax = 0.01
    debug = False
    max_steps = 100
    read_int_dirs = False
    with open("neb_input", "r") as f:
        for line in f:
            key, val = read_line_generic(line)
            if "images" in key:
                if "read" in val:
                    read_int_dirs = True
                else:
                    try:
                        nImages = int(val)
                    except:
                        pass
            if "restart" in key:
                restart_bool_str = val
                restart_bool = "true" in restart_bool_str.lower()
            if "debug" in key:
                restart_bool_str = val
                debug = "true" in restart_bool_str.lower()
            if "work" in key:
                work_dir = val.strip()
            if "initial" in key:
                initial = val.strip()
            if "final" in key:
                final = val.strip()
            if ("method" in key) and ("neb" in key):
                neb_method = val.strip()
            if ("method" in key) and ("interp" in key):
                interp_method = val.strip()
            if key[0] == "k":
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
                    max_steps = int(val.strip())
                elif ("force" in key) or ("fmax" in key):
                    fmax = float(val.strip())
            if "fmax" in key:
                fmax = float(val.strip())
    if work_dir is None:
        work_dir = os.getcwd()
    return nImages, restart_bool, work_dir, initial.strip(), final, k, neb_method, interp_method, fix_pairs, fmax, debug, max_steps, read_int_dirs


def optimizer(neb, opt="FIRE", opt_alpha=150, logfile='opt.log'):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    opt_dict = {'BFGS': BFGS, 'BFGSLineSearch': BFGSLineSearch,
                'LBFGS': LBFGS, 'LBFGSLineSearch': LBFGSLineSearch,
                'GPMin': GPMin, 'MDMin': MDMin, 'FIRE': FIRE}
    if opt in ['BFGS', 'LBFGS']:
        dyn = opt_dict[opt](neb, trajectory="neb.traj", logfile=logfile, restart='hessian.pckl', alpha=opt_alpha)
    elif opt == 'FIRE':
        dyn = opt_dict[opt](neb, trajectory="neb.traj", logfile=logfile, restart='hessian.pckl', a=(opt_alpha / 70) * 0.1)
    else:
        dyn = opt_dict[opt](neb, trajectory="neb.traj", logfile=logfile, restart='hessian.pckl')
    return dyn

def bond_constraint(atoms, indices):
    if len(indices) == 2:
        atoms.set_constraint(FixBondLength(indices[0], indices[1]))
    else:
        consts = []
        for i in range(int(np.floor(len(indices)/2.))):
            consts.append(FixBondLength(indices[2*i], indices[1+(2*i)]))
        atoms.set_constraint(consts)
    return atoms



def set_calc(exe_cmd, inputs_cmds, outfile=os.getcwd(), debug=False):
    if inputs_cmds is None:
        cmds = dup_cmds(os.path.join(outfile, "in"))
    else:
        cmds = inputs_cmds
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

# def log_stuff(message, logfname="neb_log.txt", restart=False, print_time=True, _write_type="a"):
#     if print_time:
#         message = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": " + message
#     if restart:
#         _write_type = "w"
#     with open(logfname, _write_type) as f:
#         f.write(message + "\n")

def bond_constraint(atoms, indices):
    if len(indices) > 2:
        bond_consts = []
    elif len(indices) == 2:
        atoms.set_constraint(FixBondLength(indices[0], indices[1]))
    return atoms

def endpoint_optimizer(atoms, opt="FIRE", opt_alpha=150, logfile='opt.log'):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    opt_dict = {'BFGS': BFGS, 'BFGSLineSearch': BFGSLineSearch,
                'LBFGS': LBFGS, 'LBFGSLineSearch': LBFGSLineSearch,
                'GPMin': GPMin, 'MDMin': MDMin, 'FIRE': FIRE}
    if opt in ['BFGS', 'LBFGS']:
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='tmp/hessian.pckl', alpha=opt_alpha)
    elif opt == 'FIRE':
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='tmp/hessian.pckl', a=(opt_alpha / 70) * 0.1)
    else:
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='tmp/hessian.pckl')
    return dyn

def run_endpoint(tmp_dir, fix_pairs, exe_cmd, inputs_cmds, debug=False, opter="FIRE", fmax=0.1, max_steps=50):
    atoms = read(os.path.join(tmp_dir, "POSCAR"), format="vasp")
    if debug:
        atoms.set_atomic_numbers(np.ones(len(atoms.positions)))
    atoms.pbc = [True, True, False]
    if not debug:
        bond_constraint(atoms, fix_pairs)
    calculator = set_calc(exe_cmd, inputs_cmds, outfile=tmp_dir, debug=debug)
    atoms.set_calculator(calculator)
    if os.path.exists(os.path.join(tmp_dir, "opt.log")):
        os.remove(os.path.join(tmp_dir, "opt.log"))
    dyn = endpoint_optimizer(atoms, logfile=os.path.join(tmp_dir, "opt.log"), opt=opter)
    if not debug:
        traj = Trajectory(tmp_dir + 'opt.traj', 'w', atoms, properties=['energy', 'forces'])
        dyn.attach(traj.write, interval=1)
    def write_contcar(a=atoms):
        a.write(tmp_dir + 'CONTCAR', format="vasp", direct=True)
        # insert_el(tmp_dir + 'CONTCAR')
    dyn.attach(write_contcar, interval=1)
    try:
        dyn.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        print("couldnt run??")
        print(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)


if __name__ == '__main__':
    nImages, restart_bool, work_dir, _initial, _final, k, neb_method, interp_method, fix_pairs, fmax, debug, max_steps, read_int_dirs = read_neb_inputs()
    log_stuff = lambda s: log_generic(s, work_dir, "neb", True)
    if read_int_dirs:
        log_stuff("Read existing dirs requested")
        int_dirs = get_int_dirs(work_dir)
        int_dirs_indices = get_int_dirs_indices(int_dirs)
        images = []
        images_og_dirs = []
        for i in range(len(int_dirs)):
            _dir = int_dirs[int_dirs_indices[i]]
            CONTCAR = os.path.join(_dir, "CONTCAR")
            POSCAR = os.path.join(_dir, "POSCAR")
            if os.path.exists(CONTCAR):
                log_stuff(f'Image {i}: {CONTCAR}')
                images.append(read(CONTCAR))
                images_og_dirs.append(CONTCAR)
            elif os.path.exists(POSCAR):
                log_stuff(f'Image {i}: {POSCAR}')
                images.append(read(POSCAR))
                images_og_dirs.append(POSCAR)
            else:
                log_stuff(f'No recognized input file for image {i} in {_dir}')
                raise ReferenceError(f'No recognized input file for image {i} in {_dir}')
        nImages = len(images)
    log_stuff("nImages: " + str(nImages))
    log_stuff("restart: " + str(restart_bool))
    log_stuff("work: " + str(work_dir))
    log_stuff("initial: " + str(_initial))
    log_stuff("final: " + str(_final))
    log_stuff("k: " + str(k))
    log_stuff("fmax: " + str(fmax))
    log_stuff("fix pairs: " + str(fix_pairs))
    if not debug:
        from JDFTx import JDFTx
        jdftx_exe = os.environ['JDFTx_GPU']
        exe_cmd = 'srun ' + jdftx_exe
    else:
        exe_cmd = " "
    opter = "FIRE"
    if not restart_bool:
        log_stuff("interp method: " + str(interp_method))
    else:
        log_stuff("no interp needed")
    log_stuff("neb method: " + str(neb_method))
    image_dirs = [str(i) for i in range(0, nImages)]
    os.chdir(work_dir)
    if os.path.exists("inputs"):
        inputs_cmds = read_inputs("inputs")
    else:
        inputs_cmds = dup_cmds("in")
    if not fix_pairs is None:
        if restart_bool:
            log_stuff("ignoring fix pair (images already set up)")
        else:
            log_stuff(f're-optimizing initial and final images with specified atom pairs of fixed length')
            for endpoint in [_initial, _final]:
                if os.path.exists("tmp"):
                    if os.path.isdir("tmp"):
                        shutil.rmtree("./tmp")
                os.mkdir("tmp")
                shutil.copy(endpoint, "./tmp/POSCAR")
                run_endpoint("./tmp/", fix_pairs, exe_cmd, inputs_cmds, fmax=fmax, debug=debug, opter=opter, max_steps=max_steps)
                shutil.copy("./tmp/CONTCAR", f"./{endpoint}_opted")
            _initial = _initial + "_opted"
            _final = _final + "_opted"
    if not read_int_dirs:
        if not restart_bool:
            initial = sort(read(_initial, format="vasp"))
            final = sort(read(_final, format="vasp"))
            images = [initial]
            images += [initial.copy() for i in range(nImages - 2)]
            images += [final]
            for i in range(nImages):
                if os.path.exists(str(i)) and os.path.isdir(str(i)):
                    log_stuff("resetting directory for image " + str(i))
                    shutil.rmtree("./" + str(i))
                os.mkdir(str(i))
        else:
            images = []
            for i in range(nImages):
                if os.path.exists(os.path.join(str(i), "CONTCAR")):
                    images.append(read(os.path.join(str(i), "CONTCAR"), format="vasp"))
                else:
                    images.append(read(os.path.join(str(i), "POSCAR"), format="vasp"))
    if debug:
        for im in images:
            im.set_atomic_numbers([1]*len(im.positions))
    neb = NEB(images, parallel=False, climb=True, k=k, method=neb_method)
    if not restart_bool:
        neb.interpolate(apply_constraint=True, method=interp_method)
        for i in range(nImages):
            write(os.path.join(os.path.join(work_dir, str(i)), "POSCAR"), images[i], format="vasp")
    for i in range(nImages):
        images[i].set_calculator(set_calc(exe_cmd, inputs_cmds, debug=debug, outfile=os.path.join(work_dir, str(i))))
    dyn = optimizer(neb, opt=opter, logfile='neb.log')
    traj = Trajectory('neb.traj', 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    def write_contcar(img_dir, image):
        image.write(os.path.join(img_dir, 'CONTCAR'), format="vasp", direct=True)
        insert_el(os.path.join(img_dir, 'CONTCAR'))
    for i in range(nImages):
        dyn.attach(Trajectory(os.path.join(os.path.join(work_dir, str(i)), 'opt-' + str(i) + '.traj'), 'w', images[i],
                              properties=['energy', 'forces']))
        dyn.attach(write_contcar, interval=1, img_dir=os.path.join(work_dir, str(i)), image=images[i])
    dyn.run(fmax=fmax, steps=max_steps)
    traj.close()