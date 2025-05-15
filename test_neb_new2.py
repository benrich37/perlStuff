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
    # cmd = get_exe_cmd(False, lambda p: print(p), use_srun=not debug)
    # infile = JDFTXInfile.from_file(calc_dir / "inputs", dont_require_structure=True)
    # infile["ionic-minimize"] = f"nIterations 0"
    
    # calc_setter = lambda atoms, label, restart: JDFTx(
    #     infile=infile,
    #     atoms=atoms,
    #     pseudoDir=os.environ["JDFTx_pseudo"],
    #     command=cmd,
    #     label=label,
    #     restart=restart,
    # )
    images1 = [read(calc_dir / str(i) / "POSCAR", format="vasp") for i in range(4)]
    for _ in range(3):
        images1.insert(1, images1[1].copy())
    images2 = [read(calc_dir / "0" / "POSCAR", format="vasp")]
    for _ in range(5):
        images2.append(images2[0].copy())
    images2.append(read(calc_dir / "3" / "POSCAR", format="vasp"))
    assert len(images1) == len(images2)
    tmp_neb1 = NEB(images1)
    tmp_neb2 = NEB(images2)
    tmp_neb1.interpolate(method="idpp")
    tmp_neb2.interpolate(method="idpp")
    for i, img in enumerate(tmp_neb1.images):
        posns1 = img.get_positions()
        posns2 = tmp_neb2.images[i].get_positions()
        if not all([abs(a - b) < 1e-5 for a, b in zip(posns1.flatten(), posns2.flatten())]):
            print(f"Images {i} differ")
        else:
            print(f"Images {i} are the same")



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        print(exc_info())
        exit(1)