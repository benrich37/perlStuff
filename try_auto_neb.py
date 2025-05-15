from JDFTx_new import JDFTx
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pathlib import Path

from sys import exc_info, exit, stderr
from helpers.calc_helpers import get_exe_cmd
import os
from ase.io import read, write, Trajectory
from ase.mep import NEB, AutoNEB
from ase.optimize import FIRE, LBFGS
from helpers.logx_helpers import _write_logx
from shutil import copy




def main(debug=False):
    calc_dir = Path(r"/Users/richb/Desktop/run_local/H2_H_slide")
    infile = JDFTXInfile.from_file(calc_dir / "inputs", dont_require_structure=True)
    cmd = get_exe_cmd(False, lambda p: print(p), use_srun=False)
    # calc_setter = lambda atoms, label, restart: JDFTx(
    #     infile=infile,
    #     atoms=atoms,
    #     pseudoDir=os.environ["JDFTx_pseudo"],
    #     command=cmd,
    #     label=label,
    #     restart=restart,
    # )
    # calc_setter = lambda label, atoms : JDFTx(
    #     infile=infile,
    #     pseudoDir=os.environ["JDFTx_pseudo"],
    #     command=cmd,
    #     label=label,
    #     atoms=atoms,
    # )
    def calc_setter(label, atoms):
        return JDFTx(
            infile=infile,
            pseudoDir=os.environ["JDFTx_pseudo"],
            command=cmd,
            label=label,
            atoms=atoms,
        )
    # def calc_setter(label):
    #     return JDFTx(
    #         infile=infile,
    #         pseudoDir=os.environ["JDFTx_pseudo"],
    #         command=cmd,
    #         label=label,
    #     )
    
    # def attach_calc(images):
    #     for i, image in enumerate(images):
    #         image.calc = calc_setter(image, f"{i}/jdftx", None)
    
    auto_neb_dir = calc_dir / "auto_neb"
    auto_neb_dir.mkdir(exist_ok=True)
    initial_bounds_dir = auto_neb_dir / "initial_bounds"
    initial_bounds_dir.mkdir(exist_ok=True)
    # ps = initial_bounds_dir / "start"
    # ps.mkdir(exist_ok=True)
    # ps = initial_bounds_dir / "end"
    # ps.mkdir(exist_ok=True)
    # start_restart_traj = initial_bounds_dir / "start_restart.traj"
    # end_restart_traj = initial_bounds_dir / "end_restart.traj"
    # if start_restart_traj.exists():
    #     atoms_start = read(start_restart_traj, format="traj")
    # else:
    #     atoms_start = read(calc_dir / "POSCAR_start.gjf", format="gaussian-in")
    # if end_restart_traj.exists():
    #     atoms_end = read(end_restart_traj, format="traj")
    # else:
    #     atoms_end = read(calc_dir / "POSCAR_end.gjf", format="gaussian-in")
    # atoms_start.calc = calc_setter(str(initial_bounds_dir / "start"), atoms_start.copy())
    # atoms_end.calc = calc_setter(str(initial_bounds_dir / "end"), atoms_end)
    # # atoms_start.calc = calc_setter(str(initial_bounds_dir / "start"))
    # # atoms_end.calc = calc_setter(str(initial_bounds_dir / "end"))
    # optimizer_start = FIRE(atoms_start)
    # #assert not atoms_start.calc.atoms is atoms_start
    # # optimizer_start = LBFGS(atoms_start)
    # optimizer_start.run(steps=100, fmax=0.01)
    # # write(str(auto_neb_dir / "j000.traj"), atoms_start, format="traj")
    # atoms_start.write(str(auto_neb_dir / "j000.traj"))
    # optimizer_end = FIRE(atoms_end)
    # optimizer_end.run(steps=100, fmax=0.01)
    # # write(str(auto_neb_dir / "j001.traj"), atoms_end, format="traj")
    # atoms_end.write(str(auto_neb_dir / "j001.traj"))
    def attach_calculators(images):
        print(f"Attaching calculators on {len(images)} images")
        for i, image in enumerate(images):
            image.calc = calc_setter(str(auto_neb_dir / f"{i}"), image)
            # image.calc = calc_setter(str(auto_neb_dir / f"{i}"))

    neb = AutoNEB(attach_calculators,
                  str(auto_neb_dir / "j"),
                  5,
                  7,
                  remove_rotation_and_translation=False, # Setting this to True breaks
                  parallel=False,
                  smooth_curve=True,
                  climb=True,
                  iter_folder=auto_neb_dir,
                  maxsteps=[100, 200],
                  interpolate_method='linear')
    #traj = Trajectory(str(auto_neb_dir / "neb.traj"), 'w', neb, properties=['energy', 'forces'])
    neb.run()

# def main(debug=True):
#     calc_dir = Path(r"/Users/richb/Desktop/run_local/H2_custom_images_new2")
#     infile = JDFTXInfile.from_file(calc_dir / "inputs", dont_require_structure=True)
#     cmd = get_exe_cmd(False, lambda p: print(p), use_srun=not debug)
#     calc_setter = lambda atoms, label, restart: JDFTx(
#         infile=infile,
#         atoms=atoms,
#         pseudoDir=os.environ["JDFTx_pseudo"],
#         command=cmd,
#         label=label,
#         restart=restart,
#     )
#     def attach_calc(images):
#         for i, image in enumerate(images):
#             image.calc = calc_setter(image, f"{i}/jdftx", None)
#     auto_neb_dir = calc_dir / "auto_neb"
#     auto_neb_dir.mkdir(exist_ok=True)
#     neb = AutoNEB(attach_calc, str(calc_dir / "auto_neb" / "i"), 4, 5)
#     for i in range(4):
#         copy(calc_dir / "neb" / str(i) / "_restart.traj", auto_neb_dir / f"i00{i}.traj")
#         copy(calc_dir / "neb" / str(i) / "_restart.traj", auto_neb_dir / f"i00{i}_restart.traj")
#         copy(calc_dir / "neb" / str(i) / "_restart.json", auto_neb_dir / f"i00{i}_restart.json")
#         copy(calc_dir / "neb" / str(i) / "_params.ase", auto_neb_dir / f"i00{i}_params.ase")
#     neb.run()


main()