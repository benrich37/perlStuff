# import os

# from ase import Atoms
# from ase.io import write
# from ase.calculators.lj import LennardJones
# from JDFTx_new import JDFTx
# from ase.optimize import LBFGS
# from ase.mep.autoneb import AutoNEB

# from tempfile import TemporaryDirectory

# def attach_calculators(images):
#     for image in images:
#         image.calc = LennardJones(rc=10)

# with TemporaryDirectory() as tmp_dir:
#     atoms = Atoms("ArHeNe", [(-4.484, 0., 0.),
#                              (0.198, 0., 0.),
#                              (3.286, 0., 0.)])
#     atoms.calc = LennardJones(rc=10)

#     optimizer = LBFGS(atoms)
#     optimizer.run(steps=100)
#     write(os.path.join(tmp_dir, "minimal000.traj"), atoms, format="traj")

#     atoms.set_positions([(-3.286, 0., 0.),
#                          (-0.198, 0., 0.),
#                          (4.484, 0., 0.)])
#     optimizer.run(steps=100)
#     write(os.path.join(tmp_dir, "minimal001.traj"), atoms, format="traj")

#     neb = AutoNEB(attach_calculators,
#                   os.path.join(tmp_dir, "minimal"),
#                   3,
#                   11,
#                   remove_rotation_and_translation=False, # Setting this to True breaks
#                   parallel=False,
#                   smooth_curve=True,
#                   climb=True,
#                   iter_folder=tmp_dir,
#                   maxsteps=[100, 200],
#                   interpolate_method='linear')
#     neb.run()
#     for energy in neb.get_energies():
#         print(float(energy))
#     write("finished-neb.xyz", neb.all_images)


class AClass:

    print_function = None
    log_func = lambda _, x: print(x)
    def __init__(self):
        self.print_function = print
        self._print_function = None
        self._print_function = lambda x: print(x)
        self._print_function("Hello from AClass")
        self.print_function("Hello from AClass")
        self.log_func("Hello from AClass")

AClass()