import os
import shutil
import numpy as np
from datetime import datetime
from ase.constraints import FixBondLength
from os.path import join as opj
from os.path import exists as ope
from ase.io import read, write
from scripts.traj_to_logx import log_charges, log_input_orientation, scf_str, opt_spacer
from scripts.out_to_logx import out_to_logx_str, get_atoms_from_outfile_data, get_start_line
from pathlib import Path
import copy


def log_def(s):
    print(s)


state_files = ["wfns", "eigenvals", "fillings", "fluidState"]

gbrv_15_ref = [
    "sn f ca ta sc cd sb mg b se ga os ir li si co cr pt cu i pd br k as h mn cs rb ge bi ag fe tc hf ba ru al hg mo y re s tl te ti be p zn sr n rh au hf nb c w ni cl la in v pb zr o ",
    "14. 7. 10. 13. 11. 12. 15. 10. 3. 6. 19. 16. 15. 3. 4. 17. 14. 16. 19. 7. 16. 7. 9. 5. 1. 15. 9. 9. 14. 15. 19. 16. 15. 12. 10. 16. 3. 12. 14. 11. 15. 6. 13. 6. 12. 4. 5. 20. 10. 5. 15. 11. 12. 13. 4. 14. 18. 7. 11. 13. 13. 14. 12. 6. "
]

valence_electrons = {
    'h': 1, 'he': 2,
    'li': 1, 'be': 2, 'b': 3, 'c': 4, 'n': 5, 'o': 6, 'f': 7, 'ne': 8,
    'na': 1, 'mg': 2, 'al': 3, 'si': 4, 'p': 5, 's': 6, 'cl': 7, 'ar': 8,
    'k': 1, 'ca': 2, 'sc': 2, 'ti': 2, 'v': 2, 'cr': 1, 'mn': 2, 'fe': 2, 'co': 2, 'ni': 2, 'cu': 1, 'zn': 2,
    'ga': 3, 'ge': 4, 'as': 5, 'se': 6, 'br': 7, 'kr': 8,
    'rb': 1, 'sr': 2, 'y': 2, 'zr': 2, 'nb': 1, 'mo': 1, 'tc': 2, 'ru': 2, 'rh': 1, 'pd': 0, 'ag': 1, 'cd': 2,
    'in': 3, 'sn': 4, 'sb': 5, 'te': 6, 'i': 7, 'xe': 8,
    'cs': 1, 'ba': 2, 'la': 2, 'ce': 2, 'pr': 2, 'nd': 2, 'pm': 2, 'sm': 2, 'eu': 2, 'gd': 3, 'tb': 3, 'dy': 3,
    'ho': 3, 'er': 3, 'tm': 2, 'yb': 2, 'lu': 2, 'hf': 2, 'ta': 2, 'w': 2, 're': 2, 'os': 2, 'ir': 2, 'pt': 2,
    'au': 1, 'hg': 2, 'tl': 3, 'pb': 4, 'bi': 5, 'po': 6, 'at': 7, 'rn': 8,
}

submit_gpu_perl_ref = [
    "#!/bin/bash",
    "#SBATCH -J foo",
    "#SBATCH --time=1:00:00",
    "#SBATCH -o foo.out",
    "#SBATCH -e foo.err",
    "#SBATCH -q regular_ss11",
    "#SBATCH -N 1",
    "#SBATCH -c 32",
    "#SBATCH --ntasks-per-node=4",
    "#SBATCH -C gpu",
    "#SBATCH --gpus-per-task=1",
    "#SBATCH --gpu-bind=none",
    "#SBATCH -A m4025_g\n",
    "#module use --append /global/cfs/cdirs/m4025/Software/Perlmutter/modules",
    "#module load jdftx/gpu\n",
    "export JDFTx_NUM_PROCS=1",
    "export SLURM_CPU_BIND=\"cores\"",
    "export JDFTX_MEMPOOL_SIZE=36000",
    "export MPICH_GPU_SUPPORT_ENABLED=1\n",
    "python bar/foo.py > foo.out",
    "exit 0"
]


def copy_file(file, tgt_dir, log_fn=log_def):
    shutil.copy(file, tgt_dir)
    log_fn(f"Copying {file} to {tgt_dir}")


def copy_files(src_dir, tgt_dir):
    for filename in os.listdir(src_dir):
        file_path = opj(src_dir, filename)
        if os.path.isfile(file_path):
            copy_file(file_path, tgt_dir)


def get_int_dirs_indices(int_dirs):
    ints = []
    for dirr in int_dirs:
        ints.append(int(dirr.split("/")[-1]))
    return np.array(ints).argsort()
    return np.array(ints).argsort()


def get_int_dirs(dir_path):
    int_dir_list = []
    for name in os.listdir(dir_path):
        full_path = opj(dir_path, name)
        if os.path.isdir(full_path):
            try:
                int(name)
                int_dir_list.append(full_path)
            except ValueError:
                continue
    return int_dir_list


def insert_el(filename):
    """
    Inserts elements line in correct position for Vasp 5? Good for
    nebmovie.pl script in VTST-tools package
    Args:
        filename: name of file to add elements line
    """
    with open(filename, 'r') as f:
        file = f.read()
    contents = file.split('\n')
    ele_line = contents[0]
    if contents[5].split() != ele_line.split():
        contents.insert(5, ele_line)
    with open(filename, 'w') as f:
        f.write('\n'.join(contents))


def read_inputs(work_dir, ref_struct=None):
    inpfname = opj(work_dir, "inputs")
    if os.path.exists("inputs"):
        ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump", "initial-state",
                  "coulomb-interaction", "coulomb-truncation-embed", "lattice-type", "opt", "max_steps", "fmax",
                  "optimizer", "pseudos", "logfile", "restart", "econv", "safe-mode"]
        input_cmds = {"dump": "End State"}
        with open(inpfname) as f:
            for i, line in enumerate(f):
                if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                    skip = False
                    for ig in ignore:
                        if ig in line:
                            skip = True
                    if "#" in line:
                        skip = True
                    if "ASE" in line:
                        break
                    if not skip:
                        cmd = line[:line.index(" ")]
                        rest = line.rstrip("\n")[line.index(" ") + 1:]
                        if cmd not in ignore:
                            input_cmds[cmd] = rest
        do_n_bands = False
        if "elec-n-bands" in input_cmds.keys():
            if input_cmds["elec-n-bands"] == "*":
                do_n_bands = True
        else:
            do_n_bands = True
        if do_n_bands:
            if ref_struct is None:
                ref_paths = [opj(work_dir, "POSCAR"), opj(work_dir, "CONTCAR")]
            else:
                ref_paths = [opj(work_dir, ref_struct), opj(work_dir, "CONTCAR"), opj(work_dir, "POSCAR")]
            for p in ref_paths:
                if ope(p):
                    input_cmds["elec-n-bands"] = str(get_nbands(p))
                    break
        return input_cmds
    else:
        return None


def get_nbands(poscar_fname):
    atoms = read(poscar_fname)
    count_dict = {}
    for a in atoms.get_chemical_symbols():
        if a.lower() not in count_dict.keys():
            count_dict[a.lower()] = 0
        count_dict[a.lower()] += 1
    nval = 0
    for a in count_dict.keys():
        if a in gbrv_15_ref[0].split(" "):
            idx = gbrv_15_ref[0].split(" ").index(a)
            val = (gbrv_15_ref[1].split(". "))[idx]
            count = count_dict[a]
            nval += int(val) * int(count)
        else:
            nval += int(valence_electrons[a]) * int(count_dict[a])
    return max([int(nval / 2) + 10, int((nval / 2) * 1.2)])


def dup_cmds(infile):
    lattice_line = None
    infile_cmds = {"dump": "End State"}
    ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump-name", "initial-state",
              "coulomb-interaction", "coulomb-truncation-embed"]
    with open(infile) as f:
        for i, line in enumerate(f):
            if "lattice " in line:
                lattice_line = i
            if lattice_line is not None:
                if i > lattice_line + 3:
                    if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                        skip = False
                        for ig in ignore:
                            if ig in line:
                                skip = True
                            elif line[:4] == "ion ":
                                skip = True
                        if not skip:
                            cmd = line[:line.index(" ")]
                            rest = line.rstrip("\n")[line.index(" ") + 1:]
                            if cmd not in ignore:
                                if not cmd == "dump":
                                    infile_cmds[cmd] = rest
    return infile_cmds


def copy_state_files(src, dest, log_fn=log_def):
    for f in state_files:
        if ope(opj(src, f)):
            log_fn(f"copying {f} from {src} to {dest}")
            shutil.copy(opj(src, f), dest)


def has_state_files(dirr):
    has = True
    for f in state_files:
        has = has and ope(opj(dirr, f))
    return has


def get_mtime(path_str):
    return Path(path_str).stat().st_mtime


def get_best(dir_list, f, log_fn=log_def):
    time_best = 0
    path_best = None
    for d in dir_list:
        if ope(opj(d, f)):
            time = get_mtime(opj(d, f))
            if time > time_best:
                time_best = time
                path_best = opj(d, f)
    if path_best is not None:
        return path_best
    else:
        err = f"No dirs have {f}"
        log_fn(err)
        raise ValueError(err)


def get_best_state_files(dir_list, log_fn=log_def):
    best_files = []
    for f in state_files:
        best_files.append(get_best(dir_list, f, log_fn=log_fn))
    return best_files


def copy_best_state_f(dir_list, target, log_fn):
    try:
        best = get_best_state_files(dir_list, log_fn=log_fn)
        for f in best:
            if not Path(f).parent == Path(target):
                copy_file(f, target, log_fn=log_fn)
            else:
                log_fn(f"Keeping state file {f} in {target}")
    except Exception as e:
        log_fn(e)
        pass


def remove_restart_files(dirr, log_fn=log_def):
    restart_files = ["wfns", "eigenvals", "fillings", "fluidState", "force", "hessian.pckl"]
    for f in restart_files:
        if ope(opj(dirr, f)):
            log_fn(f"removing {f} from {dirr}")
            os.remove(opj(dirr, f))


def time_to_str(t):
    if t < 60:
        print_str = f"{t:.{3}g} sec"
    elif t < 3600:
        print_str = f"{t / 60.:.{3}g} min"
    else:
        print_str = f"{t / 3600.:.{3}g} hr"
    return print_str


def atom_str(atoms, index):
    return f"{atoms.get_chemical_symbols()[index]}({index})"

def bond_str(atoms, i1, i2):
    return atom_str(atoms, i1) + "-" + atom_str(atoms, i2)


def need_sort(root):
    atoms = read(opj(root, "POSCAR"), format="vasp")
    ats = []
    dones = []
    for a in atoms.get_chemical_symbols():
        if a not in ats:
            ats.append(ats)
        elif a in dones:
            return True
        for at in ats:
            if at not in dones:
                if at != a:
                    dones.append(at)
    return False


def get_log_fn(work, calc_type, print_bool, restart=False):
    fname = opj(work, calc_type + ".iolog")
    if not restart:
        if ope(fname):
            os.remove(fname)
    return lambda s: log_generic(s, work, calc_type, print_bool)


def log_generic(message, work, calc_type, print_bool):
    message = str(message)
    if "\n" not in message:
        message = message + "\n"
    prefix = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    message = prefix + message
    log_fname = os.path.join(work, calc_type + ".log")
    if not ope(log_fname):
        with open(log_fname, "w") as f:
            f.write(prefix + "Starting\n")
            f.close()
    with open(log_fname, "a") as f:
        f.write(message)
        f.close()
    if print_bool:
        print(message)


def get_cmds(work_dir, ref_struct=None):
    os.chdir(work_dir)
    if not ope(opj(work_dir, "inputs")):
        return dup_cmds(opj(work_dir, "in"))
    else:
        return read_inputs(work_dir, ref_struct=ref_struct)


def read_line_generic(line):
    key = line.lower().split(":")[0]
    val = line.rstrip("\n").split(":")[1]
    if "#" not in key:
        if "#" in val:
            val = val[:val.index("#")]
        return key, val
    else:
        return None, None


def remove_dir_recursive(path, log_fn=log_def):
    log_fn(f"Removing directory {path}")
    for root, dirs, files in os.walk(path, topdown=False):  # topdown=False makes the walk visit subdirectories first
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)  # remove the root directory itself


def add_constraint(atoms, constraint):
    consts = atoms.constraints
    if len(consts) == 0:
        atoms.set_constraint(constraint)
    else:
        consts.append(constraint)
        atoms.set_constraint(consts)


def add_bond_constraints(atoms, indices, log_fn=log_def):
    if not len(indices) % 2 == 0:
        raise ValueError("Uneven number of indices")
    nPairs = int(len(indices) / 2)
    for i in range(nPairs):
        add_constraint(atoms, FixBondLength(indices[2 * i], indices[1 + (2 * i)]))
        cur_length = np.linalg.norm(atoms.positions[indices[0]] - atoms.positions[indices[1]])
        print_str = f"Fixed bond {atom_str(atoms, indices[0])} -"
        print_str += f" {atom_str(atoms, indices[1])} fixed to {cur_length:.{4}g} A"
        log_fn(print_str)


def _write_contcar(atoms, root):
    atoms.write(os.path.join(root, 'CONTCAR'), format="vasp", direct=True)
    insert_el(os.path.join(root, 'CONTCAR'))


def optimizer(atoms, root, opter, opt_alpha=150):
    traj = opj(root, "opt.traj")
    log = opj(root, "opt.log")
    restart = opj(root, "hessian.pckl")
    dyn = opter(atoms, trajectory=traj, logfile=log, restart=restart, a=(opt_alpha / 70) * 0.1)
    return dyn


def get_inputs_list(fname, auto_lower=True):
    inputs = []
    with open(fname, "r") as f:
        for line in f:
            if ":" in line:
                key = line.split(":")[0]
                val = line.rstrip("\n").split(":")[1]
                if "#" in val:
                    val = val[:val.index("#")]
                if auto_lower:
                    key = key.lower()
                    val = val.lower()
                if "#" not in key:
                    inputs.append(tuple([key, val]))
    return inputs


def fix_work_dir(work_dir):
    if work_dir is None:
        work_dir = os.getcwd()
    if work_dir[-1] != "/":
        work_dir += "/"
    return work_dir


def get_bond_length(atoms, indices):
    posn1 = atoms.positions[indices[0]]
    posn2 = atoms.positions[indices[1]]
    return np.linalg.norm(posn2 - posn1)


def step_bond_with_momentum(atom_pair, step_length, atoms_prev_2, atoms_prev_1):
    target_length = get_bond_length(atoms_prev_1, atom_pair) + step_length
    dir_vecs = []
    for i in range(len(atoms_prev_1.positions)):
        dir_vecs.append(atoms_prev_1[i] - atoms_prev_2[i])
    for i in range(len(dir_vecs)):
        atoms_prev_1.positions[i] += dir_vecs[i]
    dir_vec = atoms_prev_1.positions[atom_pair[1]] - atoms_prev_1.positions[atom_pair[0]]
    cur_length = np.linalg.norm(dir_vec)
    should_be_0 = target_length - cur_length
    if not np.isclose(should_be_0, 0.0):
        atoms_prev_1.positions[atom_pair[1]] += dir_vec * should_be_0 / np.linalg.norm(dir_vec)
    return atoms_prev_1


def dump_template_input(fname, template, cwd):
    dump_str = ""
    for el in template:
        dump_str += el + "\n"
    with open(opj(cwd, fname), "w") as f:
        f.write(dump_str)


def check_submit(gpu, cwd):
    if not ope(opj(cwd, "submit.sh")):
        if gpu:
            dump_template_input("submit.sh", submit_gpu_perl_ref, cwd)
        exit()


def read_pbc_val(val):
    vsplit = val.strip().split(' ')
    pbc = []
    for i in range(3):
        pbc.append("true" in vsplit[i].lower())
    return pbc


def _get_calc(exe_cmd, cmds, root, jdftx_fn, debug=False, debug_fn=None, log_fn=log_def):
    if debug:
        log_fn("Setting calc to debug calc")
        return debug_fn()
    else:
        log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {root} \n \t cmds: {cmds} \n")
        return jdftx_fn(
            executable=exe_cmd,
            pseudoSet="GBRV_v1.5",
            commands=cmds,
            outfile=root,
            ionic_steps=False
        )


def get_exe_cmd(gpu, log_fn):
    if gpu:
        _get = 'JDFTx_GPU'
    else:
        _get = 'JDFTx'
    log_fn(f"Using {_get} for JDFTx exe")
    exe_cmd = 'srun ' + os.environ[_get]
    log_fn(f"exe_cmd: {exe_cmd}")
    return exe_cmd


def read_f(dirr):
    with open(os.path.join(dirr, "Ecomponents")) as f:
        for line in f:
            if "F =" in line:
                return float(line.strip().split("=")[1])


def _write_logx(atoms, fname, dyn, maxstep, do_cell=True, do_charges=True):
    if not ope(fname):
        with open(fname, "w") as f:
            f.write("\n Entering Link 1 \n \n")
    step = dyn.nsteps
    with open(fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=do_cell))
        f.write(scf_str(atoms))
        if do_charges:
            f.write(log_charges(atoms))
        f.write(opt_spacer(step, maxstep))


def _write_opt_log(atoms, dyn, max_steps, log_fn):
    step = dyn.nsteps
    dump_str = f"Step {step}/{max_steps}: "
    dump_str += f"\t E = {atoms.get_potential_energy()}"
    try:
        dump_str += f"\t Max Force: {np.max(abs(atoms.get_forces()))}"
        dump_str += f"\t Sum of Forces: {np.sum(atoms.get_forces())}"
    except Exception as e:
        pass
    log_fn(dump_str)


def finished_logx(atoms, fname, step, maxstep, do_cell=True):
    with open(fname, "a") as f:
        f.write(log_input_orientation(atoms, do_cell=do_cell))
        f.write(scf_str(atoms))
        f.write(log_charges(atoms))
        f.write(opt_spacer(step, maxstep))
        f.write("\n Normal termination of Gaussian 16 at Fri Jul 21 12:28:14 2023.\n")


def sp_logx(atoms, fname, do_cell=True):
    if ope(fname):
        os.remove(fname)
    dump_str = "\n Entering Link 1 \n \n"
    dump_str += log_input_orientation(atoms, do_cell=do_cell)
    dump_str += scf_str(atoms)
    dump_str += log_charges(atoms)
    dump_str += "\n Normal termination of Gaussian 16 at Fri Jul 21 12:28:14 2023.\n"
    with open(fname, "w") as f:
        f.write(dump_str)


def get_count_dict(symbols):
    count_dict = {}
    for s in symbols:
        if s not in count_dict.keys():
            count_dict[s] = 1
        else:
            count_dict[s] += 1
    return count_dict


def out_to_logx(save_dir, outfile, log_fn=lambda s: print(s)):
    try:
        fname = opj(save_dir, "out.logx")
        with open(fname, "w") as f:
            f.write(out_to_logx_str(outfile))
        f.close()
    except Exception as e:
        log_fn(e)
        pass


def parse_ionpos(ionpos_fname):
    names = []
    posns = []
    coords = None
    with open(ionpos_fname, "r") as f:
        for i, line in enumerate(f):
            tokens = line.split()
            if len(tokens):
                if line.find('# Ionic positions in') >= 0:
                    coords = tokens[4]
                elif tokens[0] == "ion":
                    names.append(tokens[1])
                    posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
    return names, np.array(posns), coords


def parse_lattice(lattice_fname):
    R = np.zeros([3, 3], dtype=float)
    with open(lattice_fname, "r") as f:
        for i, line in enumerate(f):
            if i > 0:
                R[i - 1, :] = [float(x) for x in line.split()[:3]]
    return R


def parse_coords_out(ionpos_fname, lattice_fname):
    names, posns, coords = parse_ionpos(ionpos_fname)
    R = parse_lattice(lattice_fname)
    if coords != "cartesian":
        posns = np.dot(posns, R)
    return names, posns, R


def get_atoms_from_coords_out(ionpos_fname, lattice_fname):
    names, posns, R = parse_coords_out(ionpos_fname, lattice_fname)
    return get_atoms_from_outfile_data(names, posns, R)


def has_coords_out_files(dir):
    return (ope(opj(dir, "ionpos"))) and (ope(opj(dir, "lattice")))


def get_lattice_cmds(cmds, lat_iters, pbc):
    lat_cmds = copy.copy(cmds)
    lat_cmds["lattice-minimize"] = f"nIterations {lat_iters}"
    lat_cmds["latt-move-scale"] = ' '.join([str(int(v)) for v in pbc])
    return lat_cmds

def get_ionic_opt_cmds(cmds, lat_iters):
    lat_cmds = copy.copy(cmds)
    lat_cmds["ionic-minimize"] = f"nIterations {lat_iters}"
    return lat_cmds


def death_by_state(outfname, log_fn=lambda s: print(s)):
    if not ope(outfname):
        return False
    else:
        start_line = get_start_line(outfname)
        with open(outfname) as f:
            for i, line in enumerate(f):
                if i > start_line:
                    if ("bytes" in line) and ("instead of the expected" in line):
                        log_fn(line)
                        b1 = int(line.rstrip("\n").split()[4])
                        b2 = int(line.rstrip("\n").split()[-2])
                        ratio = b1/b2
                        diff = abs(ratio - 1.)
                        if diff < 0.01:
                            log_fn("You should feel fine about this - this magnitude of misalignment are caused by roundoff error")
                        return True
    return False


def check_for_restart(e, failed_before, opt_dir, log_fn=log_def):
    log_fn(e)
    if not failed_before:
        if death_by_state(opj(opt_dir, "out"), log_fn):
            log_fn("Calculation failed due to state file. Will retry without state files present")
            remove_restart_files(opt_dir, log_fn)
            # TODO: Write something to recognize misaligned hessian.pckl's so we can remove those and try again
            return True
        else:
            log_fn("Check out file - unknown issue with calculation")
            return False
    else:
        if not death_by_state(opj(opt_dir, "out"), log_fn):
            log_fn("Calculation failed without state files interfering - check out file")
        else:
            log_fn("Recognizing failure by state files when supposedly no files are present - insane")
        return False

def check_structure(structure, work, log_fn=log_def):
    use_fmt = "vasp"
    fname_out = "POSCAR"
    if "." in structure:
        suffix = structure.split(".")[1]
        if suffix is ["com", "gjf"]:
            use_fmt = "gaussian-in"
        else:
            log_fn(f"Not sure which format {structure} is in - setting format for reader to None")
            use_fmt = None
    try:
        atoms_obj = read(structure, format=use_fmt)
    except Exception as e:
        log_fn(e)
    structure = opj(work, fname_out)
    write(structure, atoms_obj, format="vasp")
    return structure


