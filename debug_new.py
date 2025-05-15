
from os import chdir
# calc_dir = r"/Users/richb/Desktop/run_local/H2_custom_images_new2"
# chdir(calc_dir)
# from neb_custom_images_use_new2 import main
# main(debug=True)


#calc_dir = r"/Users/richb/Desktop/run_local/H2_custom_images_old"
#calc_dir = r"/Users/richb/Desktop/run_local/H2_H_slide_cust3_test"
calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/HCOO_transfer_neb_par_n8_use_v3"
chdir(calc_dir)
from neb_custom_images_3 import main
main(debug=True)

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