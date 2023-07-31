import os
from ase.io.trajectory import TrajectoryReader
from os.path import join as opj
import argparse

from helpers.generic_helpers import traj_to_log_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input traj file")
    args = parser.parse_args()
    file = args.input
    file = opj(os.getcwd(), file)
    assert ".traj" in file
    traj = TrajectoryReader(file)
    with open(file[:file.index(".traj")] + "_traj.logx", "w") as f:
        f.write(traj_to_log_str(traj))
        f.close()

