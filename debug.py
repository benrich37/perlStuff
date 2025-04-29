
from os import chdir

# calc_dir = r"/media/beri9208/Expansion/scratch_backup/perl/PC3_Gr_sps/molecules/debug"
calc_dir = r"/media/beri9208/Expansion/scratch_backup/perl/test_neb_local"
chdir(calc_dir)
#from opt import main
from neb import main
main(debug=True)


