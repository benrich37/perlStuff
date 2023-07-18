import os
from os.path import join as opj
from os.path import exists as ope
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import FIRE
from ase.calculators.emt import EMT as debug_calc
from ase.neb import NEB
import numpy as np
import shutil
from neb_helpers import neb_optimizer, init_images, read_images, prep_neb
from generic_helpers import read_inputs, read_line_generic, dup_cmds, optimizer, get_int_dirs, log_generic, get_int_dirs_indices, add_bond_constraints, write_contcar
from generic_helpers import dump_template_input


neb_template = ["nImages: 10",
                "restart: True",
                "initial: POSCAR_foo",
                "final: CONTCAR_bar # comment example",
                "k: 0.2",
                "neb method: spline",
                "interp method: linear",
                "fix_pair: 22, 25",
                "fmax: 0.05",
                "max_steps: 100"]

def read_neb_inputs(fname="neb_input"):
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
        dump_template_input(fname, neb_template, os.getcwd())
        raise ValueError(f"No neb input supplied: dumping template {fname}")
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
    with open(fname, "r") as f:
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


def bond_constraint(atoms, indices):
    if len(indices) == 2:
        atoms.set_constraint(FixBondLength(indices[0], indices[1]))
    else:
        consts = []
        for i in range(int(np.floor(len(indices)/2.))):
            consts.append(FixBondLength(indices[2*i], indices[1+(2*i)]))
        atoms.set_constraint(consts)
    return atoms



def set_calc(exe_cmd, cmds, work=os.getcwd(), debug=False, debug_calc=None):
    if debug:
        return debug_calc()
    else:
        return JDFTx(
            executable=exe_cmd,
            pseudoSet="GBRV_v1.5",
            commands=cmds,
            outfile=work,
            ionic_steps=False,
            ignoreStress=True,
    )

def run_endpoint(endpoint, tmp_dir, fix_pairs, exe_cmd, inputs_cmds, debug=False, fmax=0.1, max_steps=50):
    if os.path.exists(tmp_dir):
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    shutil.copy(endpoint, opj(tmp_dir, "POSCAR"))
    run_endpoint_runner(tmp_dir, fix_pairs, exe_cmd, inputs_cmds, fmax=fmax, debug=debug, max_steps=max_steps)
    shutil.copy(opj(tmp_dir, "CONTCAR"), f"./{endpoint}_opted")


def run_endpoint_runner(tmp_dir, fix_pairs, exe_cmd, inputs_cmds, debug=False, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(tmp_dir, "POSCAR"), format="vasp")
    if debug:
        atoms.set_atomic_numbers(np.ones(len(atoms.positions)))
    atoms.pbc = [True, True, False]
    add_bond_constraints(atoms, fix_pairs)
    calculator = set_calc(exe_cmd, inputs_cmds, work=tmp_dir, debug=debug)
    atoms.set_calculator(calculator)
    if ope(opj(tmp_dir, "opt.log")):
        os.remove(opj(tmp_dir, "opt.log"))
    dyn = endpoint_optimizer(atoms)
    if not debug:
        traj = Trajectory(tmp_dir + 'opt.traj', 'w', atoms, properties=['energy', 'forces'])
        dyn.attach(traj.write, interval=1)
    dyn.attach(lambda: write_contcar(atoms, tmp_dir), interval=1)
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
        _initial = "n/a"
        _final = "n/a"
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
    if not read_int_dirs:
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
            endpoint_optimizer = lambda atoms: optimizer(atoms, opj(work_dir, "tmp"), FIRE)
            for endpoint in [_initial, _final]:
                run_endpoint(endpoint, "./tmp/", fix_pairs, exe_cmd, inputs_cmds, fmax=fmax, debug=debug, max_steps=max_steps)
            _initial = _initial + "_opted"
            _final = _final + "_opted"
    if not read_int_dirs:
        if not restart_bool:
            images = init_images(_initial, _final, nImages, work_dir, log_stuff)
        else:
            images = read_images(nImages, work_dir)
    if debug:
        for im in images:
            im.set_atomic_numbers([1]*len(im.positions))
    neb = NEB(images, parallel=False, climb=True, k=k, method=neb_method)
    prep_neb(neb, images, work_dir, lambda work: set_calc(exe_cmd, inputs_cmds, debug=debug, work=work),
             method=interp_method, restart=restart_bool)
    dyn = neb_optimizer(neb, work_dir, FIRE)
    traj = Trajectory('neb.traj', 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    for i in range(nImages):
        dyn.attach(Trajectory(opj(opj(work_dir, str(i)), 'opt-' + str(i) + '.traj'), 'w', images[i],
                              properties=['energy', 'forces']))
        dyn.attach(lambda img, img_dir: write_contcar(img, img_dir),
                   interval=1, img_dir=os.path.join(work_dir, str(i)), img=images[i])
    dyn.run(fmax=fmax, steps=max_steps)
    traj.close()