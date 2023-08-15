from os.path import join as opj
from os.path import exists as ope
from os.path import isdir as isdir
from os import mkdir as mkdir
import shutil
from ase.build.tools import sort
from ase.io import read, write
from helpers.generic_helpers import log_def, get_int_dirs, get_int_dirs_indices, read_f
import numpy as np


def neb_optimizer(neb, root, opter, opt_alpha=150):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    traj = opj(root, "neb.traj")
    log = opj(root, "neb_opt.log")
    restart = opj(root, "hessian.pckl")
    dyn = opter(neb, trajectory=traj, logfile=log, restart=restart, a=(opt_alpha / 70) * 0.1)
    return dyn


def init_images(_initial, _final, nImages, root, log_fn):
    initial = sort(read(_initial, format="vasp"))
    final = sort(read(_final, format="vasp"))
    images = [initial]
    images += [initial.copy() for i in range(nImages - 2)]
    images += [final]
    for i in range(nImages):
        if ope(opj(root, str(i))) and isdir(opj(root, str(i))):
            log_fn("resetting directory for image " + str(i))
            shutil.rmtree(opj(root, str(i)))
        mkdir(opj(root, str(i)))
    return images


def read_images(nImages, root):
    images = []
    for i in range(nImages):
        img_dir = opj(root, str(i))
        if ope(opj(img_dir, "CONTCAR")):
            images.append(read(opj(img_dir, "CONTCAR"), format="vasp"))
        else:
            images.append(read(opj(img_dir, "POSCAR"), format="vasp"))
    return images


def prep_neb(neb, images, root, set_calc_fn, pbc, method="idpp", restart=False):
    if not restart:
        neb.interpolate(apply_constraint=True, method=method)
        for i in range(len(images)):
            write(opj(opj(root, str(i)), "POSCAR"), images[i], format="vasp")
    for i in range(len(images)):
        images[i].set_calculator(set_calc_fn(opj(root, str(i))))
        images[i].pbc = pbc


def move_current_neb_to_bpdir(neb_dir, bpdir, log_fn=log_def):
    mkdir(bpdir)
    img_dirs = get_int_dirs(neb_dir)
    for path in img_dirs:
        log_fn(f"Moving {path} to {bpdir}")
        shutil.move(path, bpdir)


def is_broken_path(neb_dir, log_fn=log_def):
    gcidcs = get_good_idcs(neb_dir)
    int_dirs = get_int_dirs(neb_dir)
    int_dir_idcs = get_int_dirs_indices(int_dirs)
    for i in range(len(int_dir_idcs)):
        if not i in gcidcs:
            log_fn(f"Image {int_dirs[int_dir_idcs[i]]} does not appear to be on current transition path")
    return len(gcidcs) != len(int_dirs)


def save_broken_path(neb_dir, bpsdir, log_fn=log_def):
    if not ope(bpsdir):
        log_fn(f"Creating directory for broken paths at {bpsdir}")
        mkdir(bpsdir)
    existing_bps = get_int_dirs(bpsdir)
    bpdir = opj(bpsdir, str(len(existing_bps)))
    move_current_neb_to_bpdir(neb_dir, bpdir, log_fn=log_fn)


def get_recent_broken_path(bpsdir):
    int_dirs = get_int_dirs(bpsdir)
    int_dirs_idcs = get_int_dirs_indices(int_dirs)
    recent_broken_path = int_dirs[int_dirs_idcs[-1]]
    return recent_broken_path


def get_good_dirs(bpdir):
    int_dirs = get_int_dirs(bpdir)
    int_dirs_idcs = get_int_dirs_indices(int_dirs)
    gidcs = get_good_idcs(bpdir)
    gdirs = []
    for idx in gidcs:
        gdirs.append(int_dirs[int_dirs_idcs[idx]])
    return gdirs


def reinit_neb_from_broken_path(neb_dir, bpsdir, log_fn=log_def):
    recent_broken_path = get_recent_broken_path(bpsdir)
    gdirs = get_good_dirs(recent_broken_path)
    for i, old_img_dir in enumerate(gdirs):
        img_dir = opj(neb_dir, str(i))
        mkdir(img_dir)
        shutil.copytree(old_img_dir, img_dir)



def check_for_broken_path(neb_dir, log_fn=log_def):
    bpsdir = opj(neb_dir, "broken_paths")
    changing = is_broken_path(neb_dir, log_fn=log_fn)
    if changing:
        log_fn(f"Removing outlier images from {neb_dir}")
        save_broken_path(neb_dir, bpsdir, log_fn=log_fn)
    return changing


def get_good_idcs(neb_dir):
    int_dirs = get_int_dirs(neb_dir)
    fs = []
    for int_dir in int_dirs:
        en = read_f(int_dir)
        fs.append(en)
    gidcs = get_good_idcs_helper(fs)
    return gidcs


def get_good_idcs_helper(fs):
    imax = fs.index(np.max(fs))
    p1 = fs[:imax + 1]
    p2 = fs[imax:]
    p2_gidcs = get_good_idcs_walker(p2)
    p1_gidcs = get_good_idcs_walker(p1[::-1])
    p2_gidcs = [i + imax for i in p2_gidcs]
    p1_gidcs = [len(p1) - x - 1 for x in p1_gidcs][::-1]
    gidcs = p1_gidcs + p2_gidcs[1:]
    return gidcs


def get_good_idcs_walker(part):
    last = None
    good = []
    for i, en in enumerate(part):
        if last is None:
            last = en
            good.append(i)
        else:
            if en > last:
                if part[-1] <= last:
                    good.append(len(part) - 1)
                return good
            else:
                good.append(i)
                last = en
    return good


