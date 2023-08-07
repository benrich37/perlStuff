import os
from os.path import join as opj

from helpers.logx_helpers import out_to_logx_str

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default = 'out', help="output file")
    args = parser.parse_args()
    file = args.input
    file = opj(os.getcwd(), file)
    assert "out" in file
    with open(file + ".logx", "w") as f:
        f.write(out_to_logx_str(file))
        f.close()