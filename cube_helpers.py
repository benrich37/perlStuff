import numpy as np
from ase.io import read
from os.path import exists as ope
from os.path import join as opj
def get_start_line(outfile):
    start = None
    for i, line in enumerate(open(outfile)):
        if 'JDFTx' in line and '***' in line:
            start = i
    return start

def get_vars(outfile):
    """ Get S, R, and mu from outfile
    :param outfile:
    :return:
        - S: Grid shape (list(int))
        - R: Lattice vectors (np.ndarray(float))
        - mu: Chemical potential (float)
    :rtype: tuple
    """
    start = get_start_line(outfile)
    R = np.zeros((3, 3))
    iLine = 0
    refLine = -10
    Rdone = False
    Sdone = False
    for i, line in enumerate(open(outfile)):
        if i > start:
            if line.find('Initializing the Grid') >= 0:
                refLine = iLine
            if not Rdone:
                rowNum = iLine - (refLine + 2)
                if rowNum >= 0 and rowNum < 3:
                    R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
                if rowNum == 3:
                    Rdone = True
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
            iLine += 1
    return S, R

def get_coords_vars(outfile):
    """ get ionPos, ionNames, and R from outfile
    :param outfile: Path to output file (str)
    :return:
        - ionPos: ion positions in lattice coordinates (np.ndarray(float))
        - ionNames: atom names (list(str))
        - R: lattice vectors (np.ndarray(float))
    :rtype: tuple
    """
    start = get_start_line(outfile)
    iLine = 0
    refLine = -10
    R = np.zeros((3, 3))
    Rdone = False
    ionPosStarted = False
    ionNames = []
    ionPos = []
    for i, line in enumerate(open(outfile)):
        if i > start:
            # Lattice vectors:
            if line.find('Initializing the Grid') >= 0 and (not Rdone):
                refLine = iLine
            rowNum = iLine - (refLine + 2)
            if rowNum >= 0 and rowNum < 3:
                R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
            if rowNum == 3:
                refLine = -10
                Rdone = True
            # Coordinate system and ionic positions:
            if ionPosStarted:
                tokens = line.split()
                if len(tokens) and tokens[0] == 'ion':
                    ionNames.append(tokens[1])
                    ionPos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
                else:
                    break
            if line.find('# Ionic positions in') >= 0:
                coords = line.split()[4]
                ionPosStarted = True
            # Line counter:
            iLine += 1
    ionPos = np.array(ionPos)
    if coords != 'lattice':
        ionPos = np.dot(ionPos, np.linalg.inv(R.T))  # convert to lattice
    return ionPos, ionNames, R

def float_to_str(num):
    return f"{num:.{6}e}"

def dump_cub_inner(inner_array, S_2):
    dump_str = ""
    nLoops = int(np.floor(S_2/6.))
    nSpill = int(S_2 - (6.*nLoops))
    for i in range(nLoops):
        for j in range(6):
            dump_str += f"{float_to_str(inner_array[(6*i) + j])} "
    for i in list(range(nSpill))[::-1]:
        dump_str += f"{float_to_str(inner_array[i])} "
    dump_str += "\n"
    return dump_str

cube_types = [
    ["charge", "Electrostatic potential from Total SCF Density", "d_tot"],
    ["density"]
]

def check_expected(calc_dir, expected, check_bool):
    if not ope(expected):
        if not ope(opj(calc_dir, expected)):
            files = os.listdir(calc_dir)
            for f in files:
                if check_bool(f, expected):
                    return f
        else:
            return opj(calc_dir, expected)
    else:
        return expected

def check_file(calc_dir, fname, expected, check_bool = lambda f, e: e == f[-len(e):]):
    if fname is None:
        return check_expected(calc_dir, expected, check_bool)
    else:
        assert ope(fname)
        return fname

def check_fname(calc_dir, fname):
    if not ope(fname):
        if not ope(opj(calc_dir, fname)):
            return False
        else:
            return opj(calc_dir, fname)
    else:
        return fname

def check_multiple(calc_dir, fname, expected_list, check_bool = lambda f, e: e == f[-len(e):]):
    if not fname is None:
        fname = check_fname(calc_dir, fname)
        assert not fname is False
    else:
        for e in expected_list:
            if fname is None:
                try:
                    fname = check_file(calc_dir, fname, e, check_bool=check_bool)
                except:
                    pass
        return fname

def write_cube_helper(outfile, CONTCAR, d, cube_file_prefix=None, title_card="Electron density from Total SCF Density"):
    S, R = get_vars(outfile)
    atoms = read(CONTCAR)
    R = np.array(atoms.cell) * (1./0.529177)
    posns = atoms.positions
    nAtoms = len(posns)
    nNums = atoms.get_atomic_numbers()
    d = np.reshape(d, S)
    dump_str = f"Title card\n{title_card}\n"
    dump_str += f"{nAtoms} 0.0 0.0 0.0 \n"
    for i in range(3):
        dump_str += f"{S[i]} "
        v = R[i]/float(S[i])
        for j in range(3):
            dump_str += f"{v[j]} "
        dump_str += "\n"
    for i in range(nAtoms):
        dump_str += f"{int(nNums[i])} {float(nNums[i])} "
        posn = posns[i]
        for j in range(3):
            dump_str += f"{posn[j]} "
        dump_str += "\n"
    for i in range(S[0]):
        for j in range(S[1]):
            dump_str += dump_cub_inner(d[i,j,:], S[2])
    if cube_file_prefix is None:
        cube_file_prefix = CONTCAR
    with open(cube_file_prefix + ".cub", "w") as f:
        f.write(dump_str)