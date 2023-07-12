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
    with open("neb_input", "r") as f:
        for line in f:
            if not "#" in line:
                if "image" in line.lower().split(":")[0]:
                    nImages = line.rstrip("\n").split(":")[1]
                if "restart" in line.lower().split(":")[0]:
                    restart_bool_str = line.rstrip("\n").split(":")[1]
                    restart_bool = "true" in restart_bool_str.lower()
                if "debug" in line.lower().split(":")[0]:
                    restart_bool_str = line.rstrip("\n").split(":")[1]
                    debug = "true" in restart_bool_str.lower()
                if "work" in line.lower().split(":")[0]:
                    work_dir = line.rstrip("\n").split(":")[1].strip()
                if "initial" in line.lower().split(":")[0]:
                    initial = line.rstrip("\n").split(":")[1].strip()
                if "final" in line.lower().split(":")[0]:
                    final = line.rstrip("\n").split(":")[1].strip()
                if ("method" in line.lower().split(":")[0]) and ("neb" in line.lower().split(":")[0]):
                    neb_method = line.rstrip("\n").split(":")[1].strip()
                if ("method" in line.lower().split(":")[0]) and ("interp" in line.lower().split(":")[0]):
                    interp_method = line.rstrip("\n").split(":")[1].strip()
                if line.lower()[0] == "k":
                    k = float(line.rstrip("\n").split(":")[1].strip())
                if "fix" in line.lower().split(":")[0]:
                    lsplit = line.rstrip("\n").split(":")[1].rstrip("\n").split(",")
                    fix_pairs = []
                    for atom in lsplit:
                        try:
                            fix_pairs.append(int(atom))
                        except ValueError:
                            pass
                if "fmax" in line.lower().split(":")[0]:
                    fmax = float(line.rstrip("\n").split(":")[1].strip())
    return int(nImages), restart_bool, work_dir, initial.strip(), final, k, neb_method, interp_method, fix_pairs, fmax, debug

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


def dup_cmds(infile):
    lattice_line = None
    infile_cmds = {}
    infile_cmds["dump"] = "End State"
    ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump-name", "initial-state",
              "coulomb-interaction", "coulomb-truncation-embed"]
    with open(infile) as f:
        for i, line in enumerate(f):
            if "lattice " in line:
                lattice_line = i
            if not lattice_line is None:
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
                            if not cmd in ignore:
                                if not cmd == "dump":
                                    infile_cmds[cmd] = rest
                                # else:
                                #     infile_cmds["dump"].append(rest)
    return infile_cmds


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

def read_inputs(inpfname):
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
                if not skip:
                    cmd = line[:line.index(" ")]
                    rest = line.rstrip("\n")[line.index(" ") + 1:]
                    if not cmd in ignore:
                        input_cmds[cmd] = rest
    do_n_bands = False
    if "elec-n-bands" in input_cmds.keys():
        if input_cmds["elec-n-bands"] == "*":
            do_n_bands = True
    else:
        do_n_bands = True
    if do_n_bands:
        if os.path.exists("CONTCAR"):
            input_cmds["elec-n-bands"] = get_nbands("CONTCAR")
        else:
            input_cmds["elec-n-bands"] = get_nbands("POSCAR")
    return input_cmds

def copy_files(src_dir, tgt_dir):
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, tgt_dir)

def remove_dir_recursive(path):
    for root, dirs, files in os.walk(path, topdown=False):  # topdown=False makes the walk visit subdirectories first
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)  # remove the root directory itself

def log_stuff(message, logfname="neb_log.txt", restart=False, print_time=True, _write_type="a"):
    if print_time:
        message = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": " + message
    if restart:
        _write_type = "w"
    with open(logfname, _write_type) as f:
        f.write(message + "\n")

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
    log_stuff("starting neb", restart=True, print_time=True)
    nImages, restart_bool, work_dir, _initial, _final, k, neb_method, interp_method, fix_pairs, fmax, debug = read_neb_inputs()
    if not debug:
        from JDFTx import JDFTx
        jdftx_exe = os.environ['JDFTx_GPU']
        exe_cmd = 'srun ' + jdftx_exe
    else:
        exe_cmd = " "
    if work_dir is None:
        work_dir = os.getcwd()
    log_stuff("nImages: " + str(nImages))
    log_stuff("restart: " + str(restart_bool))
    log_stuff("work: " + str(work_dir))
    log_stuff("initial: " + str(_initial))
    log_stuff("final: " + str(_final))
    log_stuff("k: " + str(k))
    log_stuff("fmax: " + str(fmax))
    log_stuff("fix pairs: " + str(fix_pairs))
    if debug:
        opter="FIRE"
        max_steps = 5
    else:
        opter="FIRE"
        max_steps = 50
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
            log_stuff(f"re-optimizing initial and final images with specified atom pairs of fixed length")
            for endpoint in [_initial, _final]:
                if os.path.exists("tmp"):
                    if os.path.isdir("tmp"):
                        shutil.rmtree("./tmp")
                        #subprocess.run("rm -r tmp", shell=True)
                os.mkdir("tmp")
                shutil.copy(endpoint, "./tmp/POSCAR")
                #subprocess.run(f"cp {endpoint} ./tmp/POSCAR", shell=True, check=True)
                run_endpoint("./tmp/", fix_pairs, exe_cmd, inputs_cmds, fmax=fmax, debug=debug, opter=opter, max_steps=max_steps)
                shutil.copy("./tmp/CONTCAR", f"./{endpoint}_opted")
                #subprocess.run(f"cp ./tmp/CONTCAR ./{endpoint}_opted", shell=True, check=True)
                #subprocess.run("rm -r tmp",shell=True)
            _initial = _initial + "_opted"
            _final = _final + "_opted"
    if not restart_bool:
        initial = sort(read(_initial, format="vasp"))
        final = sort(read(_final, format="vasp"))
        images = [initial]
        images += [initial.copy() for i in range(nImages - 2)]
        images += [final]
        for i in range(nImages):
            if os.path.exists(str(i)) and os.path.isdir(str(i)):
                log_stuff("resetting directory for image " + str(i))
                # command = "rm -r ./" + str(i)
                # subprocess.run(command, check=True, shell=True)
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