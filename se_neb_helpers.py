import numpy as np
import os
from generic_helpers import get_int_dirs, int_dirs_indices



def get_f(path):
    with open(os.path.join(path, "Ecomponents")) as f:
        for line in f:
            if "F =" in line:
                return float(line.strip().split("=")[1])

def get_fs(work):
    int_dirs = get_int_dirs(work)
    indices = int_dirs_indices(int_dirs)
    fs = []
    for i in indices:
        fs.append(get_f(os.path.join(work, int_dirs[i])))

def has_max(fs):
    for i in range(len(fs) - 2):
        if fs[i + 2] > fs[i + 1]:
            if fs[i] < fs[i + 1]:
                return True
    return False


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
    with open(fname, "r") as f:
        for line in f:
            if not "#" in line:
                key = line.lower().split(":")[0]
                val = line.lower().rstrip("\n").split(":")[1]
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
                if line.lower()[0] == "k":
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
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    if neb_max_steps is None:
        neb_max_steps = int(max_steps / 10.)
    if work_dir is None:
        work_dir = os.getcwd()
    if work_dir[-1] != "/":
        work_dir += "/"
    return atom_pair, scan_steps, step_length, restart_idx, work_dir, follow, debug, max_steps, fmax, neb_method, interp_method, k, neb_max_steps