from os.path import join as opj
from os.path import exists as ope
import os
import shutil
from ase.build.tools import sort
from ase.io import read, write
from JDFTx import JDFTx

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
        if ope(opj(root, str(i))) and os.path.isdir(opj(root, str(i))):
            log_fn("resetting directory for image " + str(i))
            shutil.rmtree(opj(root, str(i)))
        os.mkdir(opj(root, str(i)))
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


def prep_neb(neb, images, root, set_calc_fn, pbc, method="idpp", restart=False):
    if not restart:
        neb.interpolate(apply_constraint=True, method=method)
        for i in range(len(images)):
            write(opj(opj(root, str(i)), "POSCAR"), images[i], format="vasp")
    for i in range(len(images)):
        images[i].set_calculator(set_calc_fn(opj(root, str(i))))
        images[i].pbc = pbc

