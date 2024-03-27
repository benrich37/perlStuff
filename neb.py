import os
from os.path import join as opj
from os.path import exists as ope
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import FIRE
from ase.neb import NEB
import numpy as np
import shutil
from helpers.neb_helpers import neb_optimizer, init_images, read_images, prep_neb
from helpers.generic_helpers import read_line_generic, optimizer, get_int_dirs, get_log_fn, get_int_dirs_indices, add_bond_constraints,
from helpers.generic_helpers import dump_template_input, read_pbc_val, get_cmds_dict, get_inputs_list, log_def
from helpers.calc_helpers import _get_calc, get_exe_cmd

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
    pbc = [True, True, False]
    gpu = True
    inputs = get_inputs_list(fname)
    pseudoset = "GBRV"
    for input in inputs:
        key, val = input[0], input[1]
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
        if "pbc" in key:
            pbc = read_pbc_val(val)
        if "gpu" in key:
            gpu = "true" in val.lower()
        if "pseudo" in key:
            pseudoset = val.strip()
    if work_dir is None:
        work_dir = os.getcwd()
    return nImages, restart_bool, work_dir, initial.strip(), final, k, neb_method, interp_method, fix_pairs, fmax, debug, max_steps, read_int_dirs, pbc, gpu, pseudoset


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

def run_endpoint(endpoint, tmp_dir, fix_pairs, exe_cmd, inputs_cmds, pbc, debug=False, fmax=0.1, max_steps=50):
    if os.path.exists(tmp_dir):
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    shutil.copy(endpoint, opj(tmp_dir, "POSCAR"))
    run_endpoint_runner(tmp_dir, fix_pairs, exe_cmd, inputs_cmds, pbc, fmax=fmax, debug=debug, max_steps=max_steps)
    shutil.copy(opj(tmp_dir, "CONTCAR"), f"./{endpoint}_opted")


def run_endpoint_runner(tmp_dir, fix_pairs, exe_cmd, inputs_cmds, pbc, debug=False, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(tmp_dir, "POSCAR"), format="vasp")
    if debug:
        atoms.set_atomic_numbers(np.ones(len(atoms.positions)))
    atoms.pbc = pbc
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

def read_from_int_dirs(work_dir):
    _initial = "n/a"
    _final = "n/a"
    neb_log("Read existing dirs requested")
    int_dirs = get_int_dirs(work_dir)
    int_dirs_indices = get_int_dirs_indices(int_dirs)
    images = []
    images_og_dirs = []
    for i in range(len(int_dirs)):
        _dir = int_dirs[int_dirs_indices[i]]
        CONTCAR = opj(_dir, "CONTCAR")
        POSCAR = opj(_dir, "POSCAR")
        if ope(CONTCAR):
            neb_log(f'Image {i}: {CONTCAR}')
            images.append(read(CONTCAR))
            images_og_dirs.append(CONTCAR)
        elif ope(POSCAR):
            neb_log(f'Image {i}: {POSCAR}')
            images.append(read(POSCAR))
            images_og_dirs.append(POSCAR)
        else:
            neb_log(f'No recognized input file for image {i} in {_dir}')
            raise ReferenceError(f'No recognized input file for image {i} in {_dir}')
    nImages = len(images)
    return _initial, _final, images, images_og_dirs, nImages


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
    k_float, neb_method_str = get_neb_options(schedule, log_fn=log_fn)
    neb = NEB(imgs_atoms_list, parallel=False, climb=use_ci_bool, k=k_float, method=neb_method_str)
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


if __name__ == '__main__':
    nImages, restart_bool, work_dir, _initial, _final, k, neb_method, interp_method, fix_pairs, fmax, debug, max_steps, read_int_dirs, pbc, gpu, pseudoSet = read_neb_inputs()
    neb_log = get_log_fn(work_dir, "neb", False)
    cmds = get_cmds_dict(work_dir)
    cmds = get_cmds_dict(work_dir, ref_struct="POSCAR")
    exe_cmd = get_exe_cmd(gpu, neb_log)
    get_calc = lambda root: _get_calc(exe_cmd, cmds, root, pseudoSet=pseudoSet, debug=False, log_fn=neb_log)
    if read_int_dirs:
        _initial, _final, images, image_dirs, nImages = read_from_int_dirs(work_dir)
    else:
        image_dirs = [str(i) for i in range(0, nImages)]
    neb_log("nImages: " + str(nImages))
    neb_log("restart: " + str(restart_bool))
    neb_log("work: " + str(work_dir))
    neb_log("initial: " + str(_initial))
    neb_log("final: " + str(_final))
    neb_log("k: " + str(k))
    neb_log("fmax: " + str(fmax))
    neb_log("fix pairs: " + str(fix_pairs))
    if not restart_bool:
        neb_log("interp method: " + str(interp_method))
    else:
        neb_log("no interp needed")
    neb_log("neb method: " + str(neb_method))
    os.chdir(work_dir)
    if not fix_pairs is None:
        if restart_bool:
            neb_log("ignoring fix pair (images already set up)")
        else:
            neb_log(f're-optimizing initial and final images with specified atom pairs of fixed length')
            endpoint_optimizer = lambda atoms: optimizer(atoms, opj(work_dir, "tmp"), FIRE)
            for endpoint in [_initial, _final]:
                run_endpoint(endpoint, "./tmp/", fix_pairs, exe_cmd, cmds, pbc, fmax=fmax, debug=debug, max_steps=max_steps)
            _initial = _initial + "_opted"
            _final = _final + "_opted"
    if not read_int_dirs:
        if not restart_bool:
            images = init_images(_initial, _final, nImages, work_dir, neb_log)
        else:
            images = read_images(nImages, work_dir)
    neb_dir = opj(work_dir, "neb")
    image_dirs = [opj(neb_dir, str(i)) for i in range(len(images))]
    neb = NEB(images, parallel=False, climb=True, k=k, method=neb_method)
    # At this point, if not restart, images have no interpolation yet. That is done in prep_neb
    prep_neb(neb, images, work_dir, get_calc, pbc, method=interp_method, restart=restart_bool)
    dyn = neb_optimizer(neb, work_dir, FIRE)
    traj = Trajectory('neb.traj', 'w', neb, properties=['energy', 'forces'])
    dyn.attach(traj)
    for i in range(nImages):
        dyn.attach(Trajectory(opj(image_dirs[i], 'opt-' + str(i) + '.traj'), 'w', images[i],
                              properties=['energy', 'forces']))
        dyn.attach(lambda img, img_dir: write_contcar(img, img_dir),
                   interval=1, img_dir=os.path.join(work_dir, str(i)), img=images[i])
    dyn.run(fmax=fmax, steps=max_steps)
    traj.close()