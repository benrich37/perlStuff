from JDFTx_new import JDFTx
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pathlib import Path

from sys import exc_info, exit, stderr
from helpers.calc_helpers import get_exe_cmd
from helpers.generic_helpers import get_cmds_dict, cmds_list_to_infile, cmds_dict_to_list
import os
from ase.io import read, write
from ase.optimize import FIRE
from helpers.logx_helpers import _write_logx
from time import time

def main(debug=False):
    start = time()
    calc_dir = Path(r"/Users/richb/Desktop/run_local/H2_opt_copy")
    os.chdir(calc_dir)
    cmd = get_exe_cmd(False, lambda p: print(p), use_srun=False)
    cmds = get_cmds_dict(calc_dir, ref_struct=str(calc_dir / "POSCAR"))
    cmds = cmds_dict_to_list(cmds)
    infile = cmds_list_to_infile(cmds)
    # infile["ionic-minimize"] = f"nIterations 5"
    atoms = read(calc_dir / "POSCAR", format="vasp")
    atoms.calc = JDFTx(
        label = str(calc_dir / "H2"),
        infile=infile,
        atoms=atoms,
        command=cmd,
    )
    dyn = FIRE(atoms, logfile=str(calc_dir / "test.iolog"))
    # write_logx = lambda: _write_logx(atoms, calc_dir / "opt.logx", do_cell=True)
    # dyn.attach(write_logx, interval=1)
    dyn.run(fmax=0.001, steps=100)
    print(f"Time taken: {time() - start:.2f} seconds")
    # #
    # start = time()
    # calc_dir = Path(r"/Users/richb/Desktop/run_local/H2_opt_copy2")
    # os.chdir(calc_dir)
    # cmd = get_exe_cmd(False, lambda p: print(p), use_srun=False)
    # infile = JDFTXInfile.from_file(calc_dir / "inputs", dont_require_structure=True)
    # infile["ionic-minimize"] = f"nIterations 0"
    # atoms = read(calc_dir / "POSCAR", format="vasp")
    # atoms.calc = JDFTx(
    #     label = str(calc_dir / "H2"),
    #     infile=infile,
    #     atoms=atoms,
    #     command=cmd,
    # )
    # dyn = FIRE(atoms)
    # # write_logx = lambda: _write_logx(atoms, calc_dir / "opt.logx", do_cell=True)
    # # dyn.attach(write_logx, interval=1)
    # dyn.run(fmax=0.001, steps=100)
    # print(f"Time taken: {time() - start:.2f} seconds")



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)