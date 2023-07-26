import os
from helpers.generic_helpers import get_int_dirs, get_int_dirs_indices

def get_f(path):
    with open(os.path.join(path, "Ecomponents")) as f:
        for line in f:
            if "F =" in line:
                return float(line.strip().split("=")[1])

def get_fs(work):
    int_dirs = get_int_dirs(work)
    indices = get_int_dirs_indices(int_dirs)
    fs = []
    for i in indices:
        fs.append(get_f(os.path.join(work, int_dirs[i])))

def has_max(fs):
    for i in range(len(fs) - 2):
        if fs[i + 2] > fs[i + 1]:
            if fs[i] < fs[i + 1]:
                return True
    return False


