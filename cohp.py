from helpers.cohp_helpers import run_pCOHP, parse_data, get_pCOHP_tensors
from os import getcwd
from os.path import join as opj, exists as ope
from ase.io import read
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('adsorbate_indices', metavar='N', type=int, nargs='+',
                    help='atom indices of adsorbate species')
parser.add_argument("-o", "--orb_resolved", help="resolve pCOHPs by orbitals", action="store_true")
args = parser.parse_args()
ads_indices = args.adsorbate_indices
orb_resolved = args.orb_resolved


root = getcwd()


if ope("CONTCAR.gjf"):
    atoms = read("CONTCAR.gjf", format="gaussian-in")
elif ope("CONTCAR"):
    atoms = read("CONTCAR.gjf", format="vasp")
elif ope("POSCAR.gjf"):
    atoms = read("POSCAR.gjf", format="gaussian-in")
elif ope("POSCAR"):
    atoms = read("CONTCAR.gjf", format="vasp")
else:
    raise ValueError("No poscar/contcar found")
ids = atoms.get_chemical_symbols()
nAtoms = len(ids)

proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks, orbs_dict, mu = parse_data(root=root)
tensors = get_pCOHP_tensors(proj_sabcju, E_sabcj)

idx_pairs_list = []

for idx in ads_indices:
    for i in range(nAtoms):
        if not i in ads_indices:
            idx_pairs_list.append([i, idx])

nrgs, pCOHPs, ipCOHPs, labels = run_pCOHP(root, idx_pairs_list, orb_resolved=orb_resolved)
dump_str = "Energies\t"
for i in range(len(labels)):
    dump_str += f"ipCOHP_{labels[i]}\t"
for i in range(len(labels)):
    dump_str += f"pCOHP_{labels[i]}\t"
dump_str += "\n"

def fts(num):
    return f"{num}"

for i in range(len(nrgs)):
    dump_str += fts(nrgs[i]) + "\t"
    dump_str += "\t".join([fts(v) for v in ipCOHPs[:,i]]) + "\t"
    dump_str += "\t".join([fts(v) for v in pCOHPs[:, i]]) + "\n"

savename = "o_"*orb_resolved + "pCOHP_" + "_".join(str(idx) for idx in ads_indices)
fname = opj(root, savename)
with open(fname, "w") as f:
    f.write(dump_str)



