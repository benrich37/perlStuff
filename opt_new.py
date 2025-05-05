from JDFTx_new import JDFTx
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pathlib import Path

from sys import exc_info, exit, stderr
from helpers.calc_helpers import get_exe_cmd
import os
from ase.io import read, write
from ase.optimize import FIRE
from helpers.logx_helpers import _write_logx

def main(debug=False):
    calc_dir = Path(os.getcwd())
    cmd = get_exe_cmd(False, lambda p: print(p), use_srun=not debug)
    infile = JDFTXInfile.from_file(calc_dir / "inputs", dont_require_structure=True)
    infile["ionic-minimize"] = f"nIterations 0"
    atoms = read(calc_dir / "POSCAR", format="vasp")
    atoms.calc = JDFTx(
        infile=infile,
        atoms=atoms,
        pseudoDir=os.environ["JDFTx_pseudo"],
        command=cmd,
        restart=str(calc_dir / "jdftx"),
    )
    dyn = FIRE(atoms)
    write_logx = lambda: _write_logx(atoms, calc_dir / "opt.logx", do_cell=True)
    dyn.attach(write_logx, interval=1)
    dyn.run(fmax=0.05, steps=100)
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