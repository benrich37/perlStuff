import os
import argparse
from helpers.logx_helpers import get_scan_logx_str
from os.path import join as opj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_path = os.getcwd()
    scan_logx = opj(dir_path, "scan.logx")
    with open(scan_logx, "w") as f:
        f.write(get_scan_logx_str(dir_path))
        f.close()
