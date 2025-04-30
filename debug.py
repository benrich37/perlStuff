
from os import chdir

# calc_dir = r"/media/beri9208/Expansion/scratch_backup/perl/PC3_Gr_sps/molecules/debug"
calc_dir = r"/media/beri9208/Expansion/scratch_backup/perl/test_neb_local"
#calc_dir = r"/Volumes/Ext4TB/scratch_backup/perl/PC3_Gr_sps/PC3_Gr/no_V/HCOO_transfer_neb_par"
chdir(calc_dir)
#from opt import main
from neb import main
main(debug=True)


# from pymatgen.core.units import Ha_to_eV, bohr_to_ang


# print(Ha_to_eV/bohr_to_ang)