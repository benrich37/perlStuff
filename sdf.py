
# import ase.parallel as mpi
# print(mpi.world)
# print(mpi.world.rank)
# print(mpi.world.size)
# print(mpi.world.comm)

outfile_path = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/HCOO_transfer_neb_par_n8_use_v3/relax_start/jdftx_run/out"
from pymatgen.io.jdftx.outputs import JDFTXOutfile
outfile = JDFTXOutfile.from_file(outfile_path)
for i, jstruc in enumerate(outfile.slices[-1].jstrucs):
    try:
        print(f"JDFTx: {i:>4}   t[s]: {jstruc.t_s:>8}   {outfile.etype}: {jstruc.e:6.15f} eV")
    except TypeError:
        pass
# from copy import deepcopy

# test1 = {"a": ["fdfd"]}
# test2 = test1
# test2["a"].append("fdfd")
# print(test1)
# print(test2)
# test2 = deepcopy(test2)
# test2["a"].append("fdfd")
# print(test1)
# print(test2)

# test1 = "-0.000819055280937   0.000331327812194  -0.000595520591304"
# test2 = "0.000920639140924  -0.000963550454517   0.001541543746057"
# test3 = "-0.000099890440216   0.000611806392313  -0.001060863044791"
# test1  = [float(i) for i in test1.split()]
# test2  = [float(i) for i in test2.split()]
# test3  = [float(i) for i in test3.split()]
# test = test1 + test2 + test3
# import numpy as np
# test = np.array(test)
# print(np.linalg.norm(test))
#print(test)

# test1 = {
#     "a": "a1",
#     "b": {"ba": "ba1"}
# }

# test2 = {
#     "c": "c1",
#     "b": {""}
# }


# for i in range(20):
#     print(f"{i:03d}")