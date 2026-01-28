
from os import chdir

# # calc_dir = r"/media/beri9208/Expansion/scratch_backup/perl/PC3_Gr_sps/molecules/debug"
# # calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/test_nebs/CH3Br_to_CH3OH/defaults"
# calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/HCOO_transfer_neb_par_n11_k005"
# #calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/HCOO_transfer_neb_par"
# chdir(calc_dir)
# from opt import main
# # calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/molecules/CO22H2O"
# # calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/P-H_physCO22H2O_v5"
# calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/n05V/HCOO_transfer_neb_par_n15_fix_vark1_i7_vib"
# chdir(calc_dir)
# main(debug=True)

calc_dir = r"/Users/richb/Desktop/run_local/H2_aimd"
from opt4 import main
chdir(calc_dir)
main(debug=True)

# from neb_custom_images import main
# #from neb import main
# #from neb_add_relax import main
# main(debug=True)

# test = [
#     (0,1),
#     (2,3),
#     (4,5)
# ]

# print(*test[0])

# import numpy as np
# test = np.nan
# print(isinstance(test, np.nan))


# from pymatgen.core.units import Ha_to_eV, bohr_to_ang


# print(Ha_to_eV/bohr_to_ang)