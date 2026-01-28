
from os import chdir

# calc_dir = r"/Users/richb/Desktop/run_local/H2_opt2"
calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/P-CO2_2C-H_v4"
chdir(calc_dir)
from opt4 import main
main(debug=True)

# calc_dir = r"/Users/richb/Desktop/run_local/H2_custom_images_new2"
# chdir(calc_dir)
# from neb_custom_images_use_new2 import main
# main(debug=True)


#calc_dir = r"/Users/richb/Desktop/run_local/H2_custom_images_old"
#calc_dir = r"/Users/richb/Desktop/run_local/H2_H_slide_cust3_test"
# calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/HCOO_transfer_neb_rot_n14_usenew"
# chdir(calc_dir)
# from neb_custom_images_4 import main
# main(debug=True)


# test = range(30)
# print(list(test[-5:]))

# from neb_custom_images_6 import main
# calc_dir = r"/Users/richb/Desktop/run_local/H2_diss_cust6_wrestart5"
# chdir(calc_dir)
# for i in range(7):
#     main(debug=True)

# from os import environ

# print(environ)
# print(environ["DDEC6_EXE_PATH"])

# from neb_custom_images_5 import main
# calc_dir = r"/Users/richb/Desktop/run_local/H2_diss_cust5_wrestart"
# # calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/P-H_2H2OH3O_v4_hey_neb_n5_fix"
# # calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/P-H_C-H_sC-H_tafel_f2_neb_n5_fix1"
# chdir(calc_dir)
# main(debug=True)
# # for i in range(7):
# #     main(debug=True)


# calc_dir = r"/Users/richb/Desktop/run_local/H3_bond_scan_new"
# from scan_bond_new import main
# chdir(calc_dir)
# main(debug=True)

# from opt import main
# calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/CO2_ads_neb_n10_ci_fix_vark1_TS_vib"
# calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/molecules/H2OH3O+"
# chdir(calc_dir)
# main(debug=True)


# calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/test_nebs/H2_slide/autoneb_local"
# chdir(calc_dir)
# from auto_neb import main
# main(debug=True)


# calc_dir = r"/Users/richb/Desktop/run_local/H2_diss_new"
# chdir(calc_dir)
# from test_neb_new import main
# main(debug=True)
# line = "ion H   0.394263244920353   0.444746507767707   0.439600927582060 1  Linear 0.1 0.2 0.3\n"
# print(line.split())


# from pathlib import Path
# from pymatgen.io.jdftx.inputs import JDFTXInfile
# from ase.calculators.calculator import (
#     Parameters
# )
# calc_dir = Path(r"/Users/richb/Desktop/run_local/H2_opt")
# infile = JDFTXInfile.from_file(calc_dir /  "in")

# params = Parameters(infile.as_dict())