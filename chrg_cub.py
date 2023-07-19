from cube_helpers import check_file, check_multiple, write_cube_helper
import numpy as np
import os
import argparse

def write_cube_charge(calc_dir, outfile=None, d_tot_file=None, CONTCAR=None):
    outfile = check_file(calc_dir, outfile, "out")
    d_tot_file = check_file(calc_dir, d_tot_file, "d_tot")
    CONTCAR = check_multiple(calc_dir, CONTCAR, ["CONTCAR", "POSCAR"], check_bool= lambda f, e: e in f)
    write_cube_helper(outfile, CONTCAR, np.fromfile(d_tot_file), cube_file_prefix=CONTCAR + "_chrg", title_card="Electrostatic potential from Total SCF Density")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to do create cube file")
    parser.add_argument("-o", type=str, default='out', help="output file to read (default: out)")
    parser.add_argument('-d', type=str, default='d_tot', help='name of electrostatic potential file to read (default: d_tot)')
    parser.add_argument("-c", type=str, default='CONTCAR', help="POSCAR-type file to read (default: CONTCAR)")
    args = parser.parse_args()
    calc_dir = os.getcwd()
    write_cube_charge(calc_dir, outfile = args.o, d_tot_file=args.d, CONTCAR=args.c)