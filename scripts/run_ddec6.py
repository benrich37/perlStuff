import numpy as np
from os.path import join as opj, exists as ope
from ase.io import read
from ase.units import Bohr
from ase import Atoms, Atom
from ase.data import chemical_symbols
from os import environ, chdir, getcwd
from subprocess import run
from scipy.interpolate import RegularGridInterpolator
from time import time

pbc_default = [False, False, False]
a_d_default = "/global/cfs/cdirs/m4025/Software/Perlmutter/ddec6/chargemol_09_26_2017/atomic_densities/"
exe_path = "/global/cfs/cdirs/m4025/Software/Perlmutter/ddec6/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_parallel"

# Set these to an environmental variable to override the default strings above
a_d_varname = None
exe_path_varname = None

if not a_d_varname is None:
    a_d_default = environ[a_d_varname]
if not exe_path_varname is None:
    exe_path = environ[exe_path_varname]

def add_redun_layer(d, axis):
    S_old = list(np.shape(d))
    S_new = list(np.shape(d))
    S_new[axis] += 1
    d_new = np.zeros(S_new)
    for i in range(S_old[0]):
        for j in range(S_old[1]):
            for k in range(S_old[2]):
                d_new[i,j,k] += d[i,j,k]
    axis = axis % 3
    if axis == 0:
        d_new[-1,:,:] += d[0,:,:]
    elif axis == 1:
        d_new[:,-1,:] += d[:,0,:]
    elif axis == 2:
        d_new[:,:,-1] += d[:,:,0]
    print(f"{S_old} -> {S_new}")
    return d_new, S_new

def remove_redun_layer(d, axis):
    S_new = list(np.shape(d))
    S_new[axis] -= 1
    d_new = np.zeros(S_new)
    for i in range(S_new[0]):
        for j in range(S_new[1]):
            for k in range(S_new[2]):
                d_new[i, j, k] += d[i, j, k]
    return d_new, S_new


def sum_d_periodic_grid(d, pbc):
    S_sum = list(np.shape(d))
    for i, v in enumerate(pbc):
        if v:
            S_sum[i] -= 1
    sum_d = np.sum(d[:S_sum[0], :S_sum[1], :S_sum[2]])
    return sum_d




def write_ddec6_inputs(calc_dir, outname="out", dfname="n", dupfname="n_up", ddnfname="n_dn", data_fname="density", pbc=None, a_d_path=None, max_space=0.03):
    if pbc is None:
        pbc = pbc_default
    if a_d_path is None:
        a_d_path = a_d_default
    outfile = opj(calc_dir, outname)
    non_col = True
    if ope(opj(calc_dir, dfname)):
        non_col = False
    elif not ope(opj(calc_dir, dupfname)):
        raise ValueError("Could not find electron density files")
    atoms = get_atoms(calc_dir)
    _S = get_density_shape(outfile)
    d = get_density_array(calc_dir, _S, non_col, dfname, dupfname, ddnfname)
    for i in range(3):
        d, S = add_redun_layer(d, i)
    # d, S = check_grid(d, atoms, maxspace=max_space)
    d = get_normed_d(d, atoms, outfile, pbc, S, _S)
    factors = [atoms.get_volume(), np.prod(S), np.prod(_S)]
    exps = [-1, 0, 1]
    print_all_factors(factors, exps, base=sum_d_periodic_grid(d, [True, True, True]))
    write_xsf(calc_dir, atoms, S, d, data_fname=data_fname)
    write_job_control(calc_dir, atoms, f"{data_fname}.XSF", outfile, pbc, a_d_path)

def run_ddec6(calc_dir):
    chdir(calc_dir)
    run(f"{exe_path}", shell=True, check=True)


def get_atoms(path):
    if ope(opj(path, "CONTCAR.gjf")):
        atoms = read(opj(path, "CONTCAR.gjf"), format="gaussian-in")
    elif ope(opj(path, "CONTCAR")):
        atoms = read(opj(path, "CONTCAR"), format="vasp")
    else:
        atoms = get_atoms_from_out(opj(path, "out"))
    return atoms

def get_density_shape(outfile):
    start = get_start_line(outfile)
    Sdone = False
    S = None
    for i, line in enumerate(open(outfile)):
        if i > start:
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
    if not S is None:
        return S
    else:
        raise ValueError(f"Issue finding density array shape 'S' from out file {outfile}")

def get_density_array(calc_dir, S, non_col, dfname, dupfname, ddnfname):
    if non_col:
        d = np.fromfile(opj(calc_dir, dupfname))
        d += np.fromfile(opj(calc_dir, ddnfname))
    else:
        d = np.fromfile(opj(calc_dir, dfname))
    d = d.reshape(S)
    return d


def print_all_factors(factors, exponents, base=1.0):
    nf = len(factors)
    ne = len(exponents)
    ni = ne**(nf)
    for i in range(ni):
        exps = []
        for f in range(nf):
            exp_f = int(np.floor((i % (ne**(f+1)))/(ne**f)))
            exps.append(exp_f)
        exps = [exponents[exp] for exp in exps]
        convs = [float(factors[f])**(exps[f]) for f in range(nf)]
        conv = np.prod(convs)*base
        print(str(conv) + ": [" + ", ".join([str(exp) for exp in exps]) + "]")




def interp_3d_array(array_in, S_want):
    S_cur = np.shape(array_in)
    cx = np.linspace(0, 1, S_cur[0])
    cy = np.linspace(0, 1, S_cur[1])
    cz = np.linspace(0, 1, S_cur[2])
    wx = np.linspace(0, 1, S_want[0])
    wy = np.linspace(0, 1, S_want[1])
    wz = np.linspace(0, 1, S_want[2])
    # pts = np.meshgrid(wx, wy, wz, indexing='ij', sparse=True)
    start = time()
    pts_shape = S_want
    pts_shape.append(3)
    pts = np.zeros(pts_shape)
    for i in range(S_want[0]):
        pts[i, :, :, 0] += wx[i]
    for i in range(S_want[1]):
        pts[:, i, :, 1] += wy[i]
    for i in range(S_want[1]):
        pts[:, :, i, 2] += wz[i]
    end = time()
    print(f"getting pts: {end - start}")
    interp = RegularGridInterpolator((cx, cy, cz), array_in)
    start = time()
    new_array = interp(pts)
    end = time()
    print(f"interpolating: {end - start}")
    return new_array


def adjust_grid(d, atoms, maxspace, adjust_bools):
    S_cur = np.shape(d)
    S_want = []
    for i in range(3):
        S_i = S_cur[i]
        if adjust_bools[i]:
            S_i = int(np.ceil(np.linalg.norm(atoms.cell[i])/maxspace))
        S_want.append(S_i)
    d = interp_3d_array(d, S_want)
    return d, S_want



def check_grid(d, atoms, maxspace=0.09):
    S = np.shape(d)
    spacings = [np.linalg.norm(atoms.cell[i])/np.shape(d)[i] for i in range(3)]
    adjusts = [s > maxspace for s in spacings]
    if True in adjusts:
        print(f"Density grid (spacings currently {spacings}) too coarse.\n Interpolating density to finer grid with linear interpolation")
        d, S = adjust_grid(d, atoms, maxspace, adjusts)
        print(f"New spacings: {[np.linalg.norm(atoms.cell[i])/np.shape(d)[i] for i in range(3)]}")
    return d, S


def get_normed_d(d, atoms, outfile, pbc, S, _S):
    tot_zval = get_target_tot_zval(atoms, outfile)
    pix_vol = atoms.get_volume()/(np.prod(np.shape(d))*(Bohr**3))
    sum_d = sum_d_periodic_grid(d, pbc)
    # sum_d = np.sum(d)
    d_new = (d*tot_zval/(pix_vol*sum_d*((np.prod(S))/(np.prod(_S)))))
    return d_new

def write_xsf(calc_dir, atoms, S, d, data_fname="density"):
    xsf_str = make_xsf_str(atoms, S, d, data_fname)
    xsf_fname = f"{data_fname}.XSF"
    xsf_file = opj(calc_dir, xsf_fname)
    with open(xsf_file, "w") as f:
        f.write(xsf_str)
    f.close()


def write_job_control(calc_dir, atoms, xsf_fname, outfile, pbc, a_d_path):
    nelecs = get_n_elecs(outfile)
    atom_type_count_dict = get_atom_type_count_dict(atoms)
    atom_types = list(atom_type_count_dict.keys())
    atom_type_core_elecs_dict = get_atom_type_core_elecs_dict(atom_types, outfile)
    elecs_per_atom_type_for_neutral_dict = get_elecs_per_atom_type_for_neutral_dict(atom_type_core_elecs_dict)
    elecs_for_neutral = get_elecs_for_neutral(atom_type_count_dict, elecs_per_atom_type_for_neutral_dict)
    net_charge = elecs_for_neutral - nelecs
    job_control_str = get_job_control_str(net_charge, pbc, xsf_fname, atom_type_core_elecs_dict, a_d_path)
    with open(opj(calc_dir, "job_control.txt"), "w") as f:
        f.write(job_control_str)
    f.close()


#####################

def get_target_tot_zval(atoms, outfile):
    atom_type_count_dict = get_atom_type_count_dict(atoms)
    atom_types = list(atom_type_count_dict.keys())
    Z_vals = [get_Z_val(el, outfile) for el in atom_types]
    tot_zval = 0
    for i, el in enumerate(atom_types):
        tot_zval += atom_type_count_dict[el]*Z_vals[i]
    return tot_zval



def get_job_control_str(net_charge, pbc, xsf_fname, atom_type_core_elecs_dict, a_d_path):
    dump_str = ""
    dump_str += get_net_charge_str(net_charge)
    dump_str += get_periodicity_str(pbc)
    dump_str += get_atomic_densities_str(a_d_path)
    dump_str += get_input_fname_str(xsf_fname)
    dump_str += get_charge_type_str("DDEC6")
    dump_str += get_n_core_elecs_str(atom_type_core_elecs_dict)
    return dump_str

def get_n_core_elecs_str(atom_type_core_elecs_dict):
    title = "number of core electrons"
    contents = ""
    for el in list(atom_type_core_elecs_dict.keys()):
        atomic_number = chemical_symbols.index(el)
        contents += f"{atomic_number} {int(atom_type_core_elecs_dict[el])}\n"
    return get_job_control_piece_str(title, contents)

def get_charge_type_str(charge_type):
    title = "charge type"
    contents = charge_type
    return get_job_control_piece_str(title, contents)

def get_input_fname_str(xsf_fname):
    title = "input filename"
    contents = str(xsf_fname)
    return get_job_control_piece_str(title, contents)

def get_atomic_densities_str(a_d_path):
    title = "atomic densities directory complete path"
    contents = a_d_path
    return get_job_control_piece_str(title, contents)

def get_periodicity_str(pbc):
    title = "periodicity along A, B, and C vectors"
    contents = ""
    for v in pbc:
        vstr = "false"
        if v:
            vstr = "true"
        contents += f".{vstr}.\n"
    return get_job_control_piece_str(title, contents)


def get_net_charge_str(net_charge):
    title = "net charge"
    contents = str(net_charge)
    return get_job_control_piece_str(title, contents)

def get_job_control_piece_str(title, contents):
    contents = contents.rstrip("\n")
    dump_str = f"<{title}>\n{contents}\n</{title}>\n\n"
    return dump_str

############################################

def get_elecs_for_neutral(atom_type_count_dict, elecs_per_atom_type_for_neutral_dict):
    elecs_for_neutral = 0
    for el in list(atom_type_count_dict.keys()):
        elecs_for_neutral += elecs_per_atom_type_for_neutral_dict[el]*atom_type_count_dict[el]
    return elecs_for_neutral

def get_elecs_per_atom_type_for_neutral_dict(atom_type_core_elecs_dict):
    elecs_per_atom_type_for_neutral_dict = {}
    for el in list(atom_type_core_elecs_dict.keys()):
        all_elecs = float(chemical_symbols.index(el))
        req_elecs = all_elecs - atom_type_core_elecs_dict[el]
        elecs_per_atom_type_for_neutral_dict[el] = req_elecs
    return elecs_per_atom_type_for_neutral_dict


def get_atom_type_core_elecs_dict(atom_types, outfile):
    atom_type_core_elecs_dict = {}
    for el in atom_types:
        atom_type_core_elecs_dict[el] = get_atom_type_core_elecs(el, outfile)
    return atom_type_core_elecs_dict



def get_atom_type_core_elecs(el, outfile):
    Z_val = get_Z_val(el, outfile)
    core_elecs = float(chemical_symbols.index(el)) - Z_val
    return core_elecs

def get_Z_val(el, outfile):
    start_line = get_start_line(outfile)
    reading_key = "Reading pseudopotential file"
    valence_key = " valence electrons in orbitals"
    Z_val = 0
    with open(outfile, "r") as f:
        reading = False
        for i, line in enumerate(f):
            if i > start_line:
                if reading:
                    if valence_key in line:
                        v = line.split(valence_key)[0].split(" ")[-1]
                        Z_val = int(v)
                        break
                    else:
                        continue
                else:
                    if reading_key in line:
                        fpath = line.split("'")[1]
                        fname = fpath.split("/")[-1]
                        ftitle = fname.split(".")[0].lower()
                        if el.lower() in ftitle.split("_"):
                            reading = True
                        else:
                            reading = False
    return Z_val




# readable_pseudo_types = ["upf"]
#
# def get_el_pseudo_file(el, pseudo_dir):
#     fs = listdir(pseudo_dir)
#     pseudo_file = None
#     for f in fs:
#         prefix = f.split(".")[0].lower()
#         if el.lower() in prefix.split("_"):
#             suffix = f.split(".")[-1]
#             if suffix in readable_pseudo_types:
#                 pseudo_file = f
#     return pseudo_file

def get_n_elecs(outfile):
    nelecs_key = "nElectrons: "
    with open(outfile, "r") as f:
        for line in f:
            if nelecs_key in line:
                nelec_line = line
    v = nelec_line.split(nelecs_key)[1].strip().split(" ")[0]
    nelecs = float(v)
    return nelecs


def get_atom_type_count_dict(atoms):
    count_dict = {}
    for el in atoms.get_chemical_symbols():
        if not el in count_dict:
            count_dict[el] = 0
        count_dict[el] += 1
    return count_dict

def get_start_lines(outfname, add_end=False):
    start_lines = []
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line or "Input parsed successfully" in line:
            start_lines.append(i)
        end_line = i
    if add_end:
        start_lines.append(end_line)
    return start_lines

def get_atoms_list_from_out(outfile):
    start_lines = get_start_lines(outfile, add_end=True)
    for i in range(len(start_lines) - 1):
        i_start = start_lines[::-1][i+1]
        i_end = start_lines[::-1][i]
        atoms_list = get_atoms_list_from_out_slice(outfile, i_start, i_end)
        if type(atoms_list) is list:
            if len(atoms_list):
                return atoms_list
    erstr = "Failed getting atoms list from out file"
    raise ValueError(erstr)

def get_atoms_list_from_out_slice(outfile, i_start, i_end):
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
    for i, line in enumerate(open(outfile)):
        if i > i_start and i < i_end:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                elif "R =" in line:
                    active_lattice = True
                elif "# Forces in" in line:
                    active_forces = True
                    coords_forces = line.split()[3]
                elif line.find('# Ionic positions in') >= 0:
                    coords = line.split()[4]
                    active_posns = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
                        lat_row += 1
                    else:
                        active_lattice = False
                        lat_row = 0
                elif active_posns:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'ion':
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                        if tokens[1] not in idxMap:
                            idxMap[tokens[1]] = []
                        idxMap[tokens[1]].append(j)
                        j += 1
                    else:
                        posns = np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges = np.zeros(nAtoms)
                ##########
                elif active_forces:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'force':
                        forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    else:
                        forces = np.array(forces)
                        active_forces = False
                ##########
                elif "Minimize: Iter:" in line:
                    if "F: " in line:
                        E = float(line[line.index("F: "):].split(' ')[1])
                    elif "G: " in line:
                        E = float(line[line.index("G: "):].split(' ')[1])
                elif active_lowdin:
                    if charge_key in line:
                        look = line.rstrip('\n')[line.index(charge_key):].split(' ')
                        symbol = str(look[1])
                        line_charges = [float(val) for val in look[2:]]
                        chargeDir[symbol] = line_charges
                        for atom in list(chargeDir.keys()):
                            for k, idx in enumerate(idxMap[atom]):
                                charges[idx] += chargeDir[atom][k]
                    elif "#" not in line:
                        active_lowdin = False
                        log_vars = True
                elif log_vars:
                    if np.sum(R) == 0.0:
                        R = get_input_coord_vars_from_outfile(outfile)[2]
                    if coords != 'cartesian':
                        posns = np.dot(posns, R)
                    if len(forces) == 0:
                        forces = np.zeros([nAtoms, 3])
                    if coords_forces.lower() != 'cartesian':
                        forces = np.dot(forces, R)
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
                    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
                        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(
                        nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts

def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces

def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0, momenta=None):
    atoms = Atoms()
    posns *= Bohr
    R = R.T*Bohr
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms

def get_input_coord_vars_from_outfile(outfname):
    start_line = get_start_line(outfname)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    with open(outfname) as f:
        for i, line in enumerate(f):
            if i > start_line:
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    elif tokens[0] == "lattice":
                        active_lattice = True
                    elif active_lattice:
                        if lat_row < 3:
                            R[lat_row, :] = [float(x) for x in tokens[:3]]
                            lat_row += 1
                        else:
                            active_lattice = False
                    elif "Initializing the Grid" in line:
                        break
    return names, posns, R

def get_atoms_from_out(outfile):
    atoms_list = get_atoms_list_from_out(outfile)
    return atoms_list[-1]

##


def get_start_line(outfile):
    start_lines = get_start_lines(outfile, add_end=False)
    return start_lines[-1]



def make_xsf_str(atoms, S, d, data_fname):
    dump_str = "CRYSTAL\n"
    dump_str += make_primvec_str(atoms)
    dump_str += make_primcoord_str(atoms)
    dump_str += make_datagrid_str(atoms, d, S, data_fname)
    return dump_str


def make_datagrid_str(atoms, d, S, data_fname):
    dump_str = "BEGIN_BLOCK_DATAGRID_3D\n"
    dump_str += f" DATA_from:{data_fname}.RHO\n"
    dump_str += " BEGIN_DATAGRID_3D_RHO:spin_1\n"
    _S = [str(s) for s in S]
    for i in range(3):
        dump_str += " "*(6-len(_S[i])) + _S[i]
    dump_str += "\n"
    dump_str += make_datagrid_str_lattice(atoms)
    dump_str += make_datagrid_str_dens(d, S)
    dump_str += " END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"
    return dump_str


def make_datagrid_str_dens(d, S):
    dump_str = ""
    for k in range(S[2]):
        for j in range(S[1]):
            for i in range(S[0]):
                ns = f"{d[i, j, k]:.8e}"
                dump_str += " " + ns
                # ns = f"{d[i,j,k]:.5e}"
                # dump_str += " "*(13-len(ns)) + ns
            dump_str += "\n"
    return dump_str

def make_datagrid_str_lattice(atoms):
    dump_str = ""
    origin = np.zeros(3)
    for j in range(3):
        num_str = f"{origin[j]:.8f}"
        dump_str += " "*(15-len(num_str))
        dump_str += num_str
    dump_str += "\n"
    for i in range(3):
        for j in range(3):
            num_str = f"{atoms.cell[i,j]:.8f}"
            dump_str += " "*(15-len(num_str))
            dump_str += num_str
        dump_str += "\n"
    return dump_str


def make_primvec_str(atoms):
    dump_str = "PRIMVEC\n"
    for i in range(3):
        for j in range(3):
            num_str = f"{atoms.cell[i,j]:.8f}"
            dump_str += " "*(20-len(num_str)) + num_str
        dump_str += "\n"
    return dump_str

def make_primcoord_str(atoms):
    dump_str = "PRIMCOORD\n"
    dump_str += f"   {len(atoms)} 1\n"
    at_nums = atoms.get_atomic_numbers()
    at_nums = [str(n) for n in at_nums]
    # at_nums = [f"{n:.8f}" for n in at_nums]
    posns = atoms.positions
    _posns = []
    for p in posns:
        _posns.append([])
        for i in range(3):
            _posns[-1].append(f"{p[i]:.8f}")
    for i in range(len(atoms)):
        dump_str += " "*(4-len(at_nums[i])) + at_nums[i]
        for j in range(3):
            pstr = _posns[i][j]
            dump_str += " "*(20-len(pstr))
            dump_str += pstr
        dump_str += "\n"
    return dump_str

def main():
    calc_dir = getcwd()
    write_ddec6_inputs(calc_dir , max_space=0.05)
    run_ddec6(calc_dir)

main()