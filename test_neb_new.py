from JDFTx_new import JDFTx
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pathlib import Path

from sys import exc_info, exit, stderr
from helpers.calc_helpers import get_exe_cmd
import os
from ase.io import read, write, Trajectory
from ase.mep import NEB, AutoNEB
from ase.optimize import FIRE
from helpers.logx_helpers import _write_logx
from shutil import copy

def main(debug=False):
    calc_dir = Path(os.getcwd())
    cmd = get_exe_cmd(False, lambda p: print(p), use_srun=not debug)
    infile = JDFTXInfile.from_file(calc_dir / "inputs", dont_require_structure=True)
    infile["ionic-minimize"] = f"nIterations 0"
    
    calc_setter = lambda atoms, label, restart: JDFTx(
        infile=infile,
        atoms=atoms,
        pseudoDir=os.environ["JDFTx_pseudo"],
        command=cmd,
        label=label,
        restart=restart,
    )
    # atoms_start = read(calc_dir / "POSCAR_start", format="vasp")
    # atoms_end = read(calc_dir / "POSCAR_end", format="vasp")
    # atoms_start.calc = calc_setter(atoms_start, "0/jdftx", None)
    # atoms_end.calc = calc_setter(atoms_end, "3/jdftx", None)
    # _ = atoms_start.get_forces()
    # _ = atoms_end.get_forces()
    initial = read(calc_dir / "0/jdftx_restart.traj")
    final = read(calc_dir / "3/jdftx_restart.traj")
    images = [initial]
    images += [initial.copy() for i in range(2)]
    images += [final]
    def attach_calc(images):
        for i, image in enumerate(images):
            image.calc = calc_setter(image, f"{i}/jdftx", None)
    #neb = NEB(images)
    auto_neb_dir = calc_dir / "auto_neb"
    auto_neb_dir.mkdir(exist_ok=True)
    neb = AutoNEB(attach_calc, str(calc_dir / "auto_neb" / "i"), 4, 5)
    for i in range(4):
        copy(calc_dir / str(i) / "jdftx_restart.traj", auto_neb_dir / f"i00{i}.traj")
    neb.run()
    # Interpolate linearly the potisions of the three middle images:
    # neb.interpolate()
    # # Set calculators:
    # for i, image in enumerate(images[1:3]):
    #     image.calc = calc_setter(image, f"{i+1}/jdftx", f"{i+1}/jdftx")
    # # Optimize:
    # #optimizer = FIRE(neb, trajectory='A2B.traj')
    # optimizer = FIRE(neb)
    # optimizer.attach(
    #     Trajectory(calc_dir / "neb.traj", 'w', neb, properties=['energy', 'forces']),
    #     interval=1,
    # )
    # optimizer.run(fmax=0.04)
    # dyn = optimizer(atoms, calc_dir, FIRE)
    # traj = Trajectory(opj(root, "opt.traj"), 'w', atoms, properties=['energy', 'forces', 'charges'])
    # logx = opj(root, "opt.logx")
    # 
    # write_contcar = lambda: _write_contcar(atoms, root)
    # write_opt_log = lambda: _write_opt_iolog(atoms, dyn, max_steps, log_fn)
    # dyn.attach(traj.write, interval=1)
    # dyn.attach(write_contcar, interval=1)
    # dyn.attach(write_logx, interval=1)
    # dyn.attach(write_opt_log, interval=1)
    # atoms.get_properties(["energy", "stress", "forces", "charges", "eigenvalues"])



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)