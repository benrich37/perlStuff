import os
from itertools import product
from os.path import join as opj, exists as ope

import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti
from ase.io import read
from numba import jit

from helpers.jparse_helpers import get_atom_orb_labels_dict, get_kfolding, parse_complex_bandfile, orbs_idx_dict, \
    parse_kptsfile, get_mu


def atom_idx_to_key_map(atoms):
    el_counter_dict = {}
    idx_to_key_map = []
    els = atoms.get_chemical_symbols()
    for i, el in enumerate(els):
        if not el in el_counter_dict:
            el_counter_dict[el] = 0
        el_counter_dict[el] += 1
        idx_to_key_map.append(f"{el} #{el_counter_dict[el]}")
    return idx_to_key_map


def get_min_dist(posn1, posn2, cell, pbc):
    dists = []
    xrs = []
    for x in pbc:
        if x:
            xrs.append([-1, 0, 1])
        else:
            xrs.append([0])
    for a in xrs[0]:
        for b in xrs[1]:
            for c in xrs[2]:
                _p2 = posn2 + (a*cell[0]) + (b*cell[1]) + (c*cell[2])
                dists.append(np.linalg.norm(posn1 - _p2))
    return np.min(dists)


def get_pair_idcs(atoms, pbc, atomic_radii_dict, tol=0.1):
    pairs = []
    els = atoms.get_chemical_symbols()
    posns = atoms.positions
    for i, el1 in enumerate(els):
        for j, el2 in enumerate(els):
            if i > j:
                cutoff = atomic_radii_dict[el1] + atomic_radii_dict[el2]
                if get_min_dist(posns[i], posns[j], atoms.cell, pbc) <= cutoff + tol:
                    pairs.append([i, j])
    return pairs


def norm_projs(proj_kju):
    nStates = np.shape(proj_kju)[0]
    nBands = np.shape(proj_kju)[1]
    nProj = np.shape(proj_kju)[2]
    u_sums = np.zeros(nProj)
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                c = abs(proj_kju[k, j, u])**2
                u_sums[u] += c
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                proj_kju[k, j, u] *= (1/np.sqrt(u_sums[u]))
    return proj_kju


def get_jnorms(proj_kju):
    nStates = np.shape(proj_kju)[0]
    nBands = np.shape(proj_kju)[1]
    nProj = np.shape(proj_kju)[2]
    j_sums = np.zeros(nBands)
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                c = abs(proj_kju[k,j,u])**2
                j_sums[j] += c
    for j in range(len(j_sums)):
        j_sums[j] = np.sqrt(j_sums[j])
    return j_sums


def get_knorms(proj_kju):
    nStates = np.shape(proj_kju)[0]
    nBands = np.shape(proj_kju)[1]
    nProj = np.shape(proj_kju)[2]
    k_sums = np.zeros(nStates)
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                c = abs(proj_kju[k,j,u])**2
                k_sums[k] += c
    for k in range(len(k_sums)):
        k_sums[k] = np.sqrt(k_sums[k])
    return k_sums


@jit(nopython=True)
def get_P_uvjsabc_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin):
    for u in range(nProj):
        for v in range(nProj):
            for j in range(nBands):
                for a in range(nKa):
                    for b in range(nKb):
                        for c in range(nKc):
                            for s in range(nSpin):
                                t1 = proj_sabcju[s,a,b,c,j,u]
                                t2 = proj_sabcju[s,a,b,c,j,v]
                                P_uvjsabc[u, v, j, s, a, b, c] = np.conj(t1)*t2
    return P_uvjsabc


def get_pdos_weights(root, orb_idcs, proj_tensor=None, decompose=False):
    if proj_tensor is None:
        proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks, orbs_dict, mu = parse_data(root=root)
        proj_tensor = proj_sabcju
    pshape = np.shape(proj_tensor)
    wshape = [pshape[0], pshape[1], pshape[2], pshape[3], pshape[4]]
    if decompose:
        weights = []
        for u in orb_idcs:
            weights.append(get_pdos_weights(root, [u], proj_tensor=proj_tensor, decompose=False))
        return weights
    else:
        weights = np.zeros(wshape)
        for u in orb_idcs:
            weights += abs(proj_tensor[:,:,:,:,:,u])**2
    return weights


def get_dos(root, Erange=None):
    atoms = read(opj(root, "CONTCAR"), format="vasp")
    proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks, orbs_dict, mu = parse_data(root=root)
    if Erange is None:
        dE = get_de(E_sabcj)
        Erange = np.arange(np.min(E_sabcj) - dE*100, np.max(E_sabcj) + dE*100, dE)
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    dos = np.zeros(np.shape(Erange))
    for s in range(nSpin):
        dos += lti(atoms.cell, E_sabcj[s], Erange)
    return dos, Erange


def get_pdos(root, atom_idx, decompose=False, Erange=None):
    proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks, orbs_dict, mu = parse_data(root=root)
    atoms = read(opj(root, "CONTCAR"), format="vasp")
    orb_idcs = orbs_dict[atom_idx_to_key_map(atoms)[atom_idx]]
    if Erange is None:
        dE = get_de(E_sabcj)
        Erange = np.arange(np.min(E_sabcj) - dE*100, np.max(E_sabcj) + dE*100, dE)
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    weights = get_pdos_weights(root, orb_idcs, proj_tensor=proj_sabcju, decompose=decompose)
    if decompose:
        pdos = []
        for u in orb_idcs:
            _pdos = np.zeros(np.shape(Erange))
            for s in range(nSpin):
                _pdos += lti(atoms.cell, E_sabcj[s], Erange, weights=weights[u][s])
            pdos.append(_pdos)
    else:
        pdos_sum = np.zeros(np.shape(Erange))
        pdos_dif = np.zeros(np.shape(Erange))
        for s in range(nSpin):
            _p = lti(atoms.cell, E_sabcj[s], Erange, weights=weights[s])
            pdos_sum += _p
            pdos_dif += _p*(1-(2*s))
    return Erange, pdos_sum


def get_P_uvjsabc(proj_sabcju):
    shape = np.shape(proj_sabcju)
    print(shape)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nProj = shape[5]
    P_uvjsabc = np.zeros([nProj, nProj, nBands, nSpin, nKa, nKb, nKc], dtype=complex)
    return get_P_uvjsabc_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin)


@jit(nopython=True)
def get_H_uvsabc_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin):
    for u in range(nProj):
        for v in range(nProj):
            for j in range(nBands):
                for s in range(nSpin):
                    for a in range(nKa):
                        for b in range(nKb):
                            for c in range(nKc):
                                H_uvsabc[u, v, s, a, b, c] += P_uvjsabc[u,v,j,s,a,b,c]*E_sabcj[s,a,b,c,j]
    return H_uvsabc


def get_H_uvsabc(P_uvjsabc, E_sabcj):
    shape = np.shape(P_uvjsabc)
    nProj = shape[0]
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    H_uvsabc = np.zeros([nProj, nProj, nSpin, nKa, nKb, nKc], dtype=complex)
    return get_H_uvsabc_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin)


@jit(nopython=True)
def get_pCOHP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                uv_sum += np.real(P_uvjsabc[u, v, j, s, a, b, c]*H_uvsabc[u, v, s, a, b, c])*wk_sabc[s,a,b,c]
                        pCOHP_sabcj[s, a, b, c, j] += uv_sum
    return pCOHP_sabcj


def get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, orbs_u, orbs_v, wk_sabc=None):
    shape = np.shape(P_uvjsabc)
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    pCOHP_sabcj = np.zeros([nSpin, nKa, nKb, nKc, nBands])
    if wk_sabc is None:
        wk_sabc = np.ones([nSpin, nKa, nKb, nKc])
    return get_pCOHP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj)


def get_pCOHP_E(pCOHP_sabcj, Erange, atoms, E_sabcj):
    shape = np.shape(pCOHP_sabcj)
    nSpin = shape[0]
    pCOHP_E = np.zeros(np.shape(Erange))
    for s in range(nSpin):
        pCOHP_E += lti(atoms.cell, E_sabcj[s], Erange, weights=pCOHP_sabcj[s])
    return pCOHP_E


def bin_pCOHP_scatter(Erange, dE, nrgs, pCOHP_scatter):
    output = np.zeros(np.shape(Erange))
    for i, nrg in enumerate(nrgs):
        iE = int(np.floor((nrg - Erange[0])/dE))
        output[iE] += pCOHP_scatter[i]
    return output


def get_de(E_sabcj, spandiv=4):
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    by_band = []
    for s in range(nSpin):
        for j in range(nBands):
            vals = []
            for a in range(nKa):
                for b in range(nKb):
                    for c in range(nKc):
                        vals.append(E_sabcj[s,a,b,c,j])
            by_band.append(vals)
    spanEs = []
    for vgroup in by_band:
        idcs = np.argsort(vgroup)
        vals = [vgroup[idx] for idx in idcs]
        spanE = vals[-1] - vals[0]
        spanEs.append(spanE)
        mindE = 100
        for i in range(len(vals) - 1):
            mindE = min(mindE, vals[i+1]-vals[i])
    return min(spanEs)/spandiv


def interp_ipcohp(sidcs, midcs, ipCOHP, Erange):
    out = np.zeros(np.shape(Erange))
    ipCOHP = [ipCOHP[idx] for idx in sidcs]
    last = 0
    for i, midx in enumerate(midcs):
        out[last:midx] += ipCOHP[i]
        last = midx
    out[last:] += ipCOHP[-1]
    return out


def interp_icohps(nrgs, ipCOHPs, Erange, dE):
    sidcs = np.argsort(nrgs)
    nrgs = [nrgs[idx] for idx in sidcs]
    midcs = []
    for i, nrg in enumerate(nrgs):
        iE = int(np.floor((nrg - Erange[0])/dE))
        midcs.append(iE)
    interped_ipCOHPs = []
    for ipCOHP in ipCOHPs:
        interped_ipCOHPs.append(interp_ipcohp(sidcs, midcs, ipCOHP, Erange))
    return interped_ipCOHPs


def get_pCOHP(E_sabcj, P_uvjsabc, H_uvsabc, orbs_u, orbs_v):
    shape = np.shape(P_uvjsabc)
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    nrgs = []
    vals = []
    for j in range(nBands):
        for s in range(nSpin):
            for a in range(nKa):
                for b in range(nKb):
                    for c in range(nKc):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                uv_sum += np.real(H_uvsabc[u,v,s,a,b,c]*P_uvjsabc[u,v,j,s,a,b,c])
                        eig = E_sabcj[s,a,b,c,j]
                        vals.append(uv_sum)
                        nrgs.append(eig)
    vals = np.array(vals)
    nrgs = np.array(nrgs)
    return nrgs, vals


def int_pCOHP(vals, occ_sabcj, wk_sabc):
    integ = [0]
    shape = np.shape(occ_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    idx = 0
    for j in range(nBands):
        for s in range(nSpin):
            for a in range(nKa):
                for b in range(nKb):
                    for c in range(nKc):
                        v = vals[idx]*wk_sabc[s,a,b,c]*occ_sabcj[s,a,b,c,j]
                        integ.append(v + integ[-1])
    return np.array(integ[1:])


def get_pCOHP_tensors(proj_sabcju, E_sabcj):
    print(f"Constructing projection tensor ...")
    P_uvjsabc = get_P_uvjsabc(proj_sabcju)
    print(f"Constructing Hamiltonian tensor ...")
    H_uvsabc = get_H_uvsabc(P_uvjsabc, E_sabcj)
    return P_uvjsabc, H_uvsabc


def run_pCOHP(root, idx_pairs, _tensors=None, gamma=False, kidcs=None, orb_resolved=False):
    print(f"Parsing Data ...")
    proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks, orbs_dict, mu = parse_data(root=root, gamma=gamma, kidcs=kidcs)
    if _tensors is None:
        _tensors = get_pCOHP_tensors(proj_sabcju, E_sabcj)
    P_uvjsabc = _tensors[0]
    H_uvsabc = _tensors[1]
    atoms = read(opj(root, "CONTCAR"), format="vasp")
    kmap = atom_idx_to_key_map(atoms)
    ipCOHPs = []
    pCOHPs = []
    labels = []
    orb_labels_dict = get_atom_orb_labels_dict(root)
    print(f"Running through idx pairs ...")
    for i, p in enumerate(idx_pairs):
        k1 = kmap[p[0]]
        k2 = kmap[p[1]]
        print(f"pCOHP {k1},{k2} ({i+1}/{len(idx_pairs)})")
        if not orb_resolved:
            label = f"{k1},{k2}"
            labels.append(label)
            nrgs, _pCOHP = get_pCOHP(E_sabcj, P_uvjsabc, H_uvsabc, orbs_dict[k1], orbs_dict[k2])
            _ipCOHP = int_pCOHP(_pCOHP, occ_sabcj, wk_sabc)
            pCOHPs.append(_pCOHP)
            ipCOHPs.append(_ipCOHP)
        else:
            sym1 = k1.split("#")[0]
            sym2 = k2.split("#")[0]
            orblabels1 = orb_labels_dict[sym1]
            orblabels2 = orb_labels_dict[sym2]
            for ui, u in enumerate(orbs_dict[k1]):
                for vi, v in enumerate(orbs_dict[k2]):
                    label = f"{k1}({orblabels1[ui]}),{k2}({orblabels2[vi]})"
                    labels.append(label)
                    nrgs, _pCOHP = get_pCOHP(E_sabcj, P_uvjsabc, H_uvsabc, [u], [v])
                    _ipCOHP = int_pCOHP(_pCOHP, occ_sabcj, wk_sabc)
                    pCOHPs.append(_pCOHP)
                    ipCOHPs.append(_ipCOHP)
    pCOHPs = np.array(pCOHPs)
    ipCOHPs = np.array(ipCOHPs)
    return nrgs, pCOHPs, ipCOHPs, labels


def run_pCOHP_plot(root, idx_pairs, tet_int=False, gamma=False):
    print(f"Parsing Data")
    proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks, orbs_dict, mu = parse_data(root=root, gamma=gamma)
    _tensors = get_pCOHP_tensors(proj_sabcju, E_sabcj)
    nrgs, pCOHPs, ipCOHPs, labels = run_pCOHP(root, idx_pairs, _tensors=_tensors)
    print(f"Making data plottable ...")
    dE = get_de(E_sabcj)
    print(f"Optimal dE calculated as {dE:.5e}")
    Erange = np.arange(np.min(nrgs) - dE*100, np.max(nrgs) + dE*100, dE)
    pCOHPs_plot = []
    ipCOHPs_plot = interp_icohps(nrgs, ipCOHPs, Erange, dE)
    if not tet_int:
        for i, _pCOHP in enumerate(pCOHPs):
            pCOHP_plot = bin_pCOHP_scatter(Erange, dE, nrgs, _pCOHP)
            pCOHPs_plot.append(pCOHP_plot)
    else:
        P_uvjsabc = _tensors[0]
        H_uvsabc = _tensors[1]
        atoms = read(opj(root, "CONTCAR"), format="vasp")
        kmap = atom_idx_to_key_map(atoms)
        pCOHPs_plot = np.zeros([len(idx_pairs), len(Erange)])
        for i, p in enumerate(idx_pairs):
            pCOHP_sabcj = get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, orbs_dict[kmap[p[0]]], orbs_dict[kmap[p[1]]])
            pCOHPs_plot[i] += get_pCOHP_E(pCOHP_sabcj, Erange, atoms, E_sabcj)
    return Erange, pCOHPs_plot, ipCOHPs_plot


def get_atoms_w_ipcohp_as_charges(root, central_idx, scale_up=True, gamma=False, kidcs=None):
    atoms = read(opj(root, "CONTCAR.gjf"), format="gaussian-in")
    look_pairs = []
    for i in range(len(atoms)):
        if not i == central_idx:
            look_pairs.append([i, central_idx])
    nrgs, pCOHPs, ipCOHPs, labels = run_pCOHP(root, look_pairs, gamma=gamma, kidcs=kidcs)
    _ipcohps = np.zeros(len(atoms))
    for i in range(len(atoms)):
        if i == central_idx:
            pass
        else:
            if i < central_idx:
                j = i
            elif i > central_idx:
                j = i - 1
            _ipcohps[i] += ipCOHPs[j][-1]
    if scale_up:
        ipcohp_max = np.max(np.abs(_ipcohps))
        scalar = 1/ipcohp_max
        print(f"Scaling ipCOHP by {scalar:.5e}")
        _ipcohps *= scalar
    atoms.set_initial_charges(_ipcohps)
    return atoms


logx_init_str = "\n Entering Link 1 \n \n"
logx_finish_str = " Normal termination of Gaussian 16"


def log_input_orientation(atoms, do_cell=False):
    dump_str = "                          Input orientation:                          \n"
    dump_str += " ---------------------------------------------------------------------\n"
    dump_str += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    dump_str += " Number     Number       Type             X           Y           Z\n"
    dump_str += " ---------------------------------------------------------------------\n"
    at_ns = atoms.get_atomic_numbers()
    at_posns = atoms.positions
    nAtoms = len(at_ns)
    for i in range(nAtoms):
        dump_str += f" {i+1} {at_ns[i]} 0 "
        for j in range(3):
            dump_str += f"{at_posns[i][j]} "
        dump_str += "\n"
    if do_cell:
        cell = atoms.cell
        for i in range(3):
            dump_str += f"{i + nAtoms + 1} -2 0 "
            for j in range(3):
                dump_str += f"{cell[i][j]} "
            dump_str += "\n"
    dump_str += " ---------------------------------------------------------------------\n"
    return dump_str


def scf_str(atoms, e_conv=(1/27.211397)):
    E = 0
    return f"\n SCF Done:  E =  {E*e_conv}\n\n"


def log_charges(atoms):
    charges = atoms.get_initial_charges()
    nAtoms = len(atoms.positions)
    symbols = atoms.get_chemical_symbols()
    dump_str = " **********************************************************************\n\n"
    dump_str += "            Population analysis using the SCF Density.\n\n"
    dump_str = " **********************************************************************\n\n Mulliken charges:\n    1\n"
    for i in range(nAtoms):
        dump_str += f"{int(i+1)} {symbols[i]} {charges[i]} \n"
    dump_str += f" Sum of Mulliken charges = {np.sum(charges)}\n"
    return dump_str


def sp_logx(atoms, fname, do_cell=True):
    if ope(fname):
        os.remove(fname)
    dump_str = logx_init_str
    dump_str += log_input_orientation(atoms, do_cell=do_cell)
    dump_str += scf_str(atoms)
    dump_str += log_charges(atoms)
    dump_str += logx_finish_str
    with open(fname, "w") as f:
        f.write(dump_str)


def dump_cohp_logx_by_kidx(root, central_idx, scale_up=True):
    kfolding = get_kfolding(opj(root, "out"))
    for a,b,c in product(*[range(k) for k in kfolding]):
        kidcs = [a,b,c]
        savename = f"cohp_{central_idx}_{a}{b}{c}"
        dump_cohp_logx(root, central_idx, scale_up=scale_up, kidcs=kidcs, savename=savename)


def dump_cohp_logx(root, central_idx, scale_up=True, gamma=False, kidcs=None, savename=None):
    atoms = get_atoms_w_ipcohp_as_charges(root, central_idx, scale_up=scale_up, gamma=gamma, kidcs=kidcs)
    if savename is None:
        savename = f"cohp_{central_idx}"
    if gamma:
        savename += "_g"
    savename += ".logx"
    fname = opj(root, savename)
    sp_logx(atoms, fname, do_cell=True)


def parse_data(root=None, bandfile="bandProjections", kPtsfile="kPts", eigfile="eigenvals", fillingsfile="fillings", outfile="out", gamma=False, kidcs=None):
    """
    :param bandfile: Path to BandProjections file (str)
    :param gvecfile: Path to Gvectors file (str)
    :param eigfile: Path to eigenvalues file (str)
    :param guts: Whether to data not directly needed by main functions (Boolean)
    :return:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of band functions (integer)
        - nProj: the number of band projections (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure (list(int))
        - wk: A list of weight factors for each k-point (list(float))
        - k_points: A list of k-points (given as 3 floats) for each k-point. (list(list(float))
        - E: nStates by nBands array of KS eigenvalues (np.ndarray(float))
        *- iGarr: A list of numpy arrays for the miller indices of each G-vector used in the
                  expansion of each state (list(np.ndarray(int)))
    :rtype: tuple
    """
    if not root is None:
        bandfile = opj(root, bandfile)
        kPtsfile = opj(root, kPtsfile)
        eigfile = opj(root, eigfile)
        fillingsfile = opj(root, fillingsfile)
        outfile = opj(root, outfile)
    proj_kju, nStates, nBands, nProj, nSpecies, nOrbsPerAtom = parse_complex_bandfile(bandfile)
    orbs_dict = orbs_idx_dict(outfile, nOrbsPerAtom)
    wk, ks, nStates = parse_kptsfile(kPtsfile)
    wk = np.array(wk)
    ks = np.array(ks)
    kfolding = get_kfolding(outfile)
    nK = int(np.prod(kfolding))
    nSpin = int(nStates/nK)
    wk = wk.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2]])
    ks = ks.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2], 3])
    E = np.fromfile(eigfile)
    Eshape = [nSpin, kfolding[0], kfolding[1], kfolding[2], nBands]
    E_sabcj = E.reshape(Eshape)
    fillings = np.fromfile(fillingsfile)
    occ_sabcj = fillings.reshape(Eshape)
    # Normalize such that sum(occ_kj) = nelec
    # occ_sabcj *= (1/nK)
    # Normalize such that sum_jk(<u|jk><jk|u>) = 1
    proj_kju = norm_projs(proj_kju)
    proj_shape = Eshape
    proj_shape.append(nProj)
    proj_flat = proj_kju.flatten()
    proj_sabcju = proj_flat.reshape(proj_shape)
    mu = get_mu(outfile)
    print(f"proj {np.shape(proj_sabcju)}")
    if not kidcs is None:
        abc = []
        for i in range(3):
            abc.append([kidcs[i], kidcs[i]+1])
        proj_sabcju = proj_sabcju[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :, :]
        E_sabcj = E_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        occ_sabcj = occ_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        wk = np.ones(np.shape(occ_sabcj[:,:,:,:,0]))*0.5
        ks = [np.zeros([3])]
    elif gamma:
        abc = []
        for i in range(3):
            kfi = kfolding[i]
            kfi_0 = int(np.ceil(kfi/2.)-1)
            kfi_p = 1
            if kfi % 2 == 0:
                kfi_p += 1
            abc.append([kfi_0, kfi_0+kfi_p])
        proj_sabcju = proj_sabcju[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :, :]
        E_sabcj = E_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        occ_sabcj = occ_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        wk = np.ones(np.shape(occ_sabcj[:,:,:,:,0]))*0.5
        ks = [np.zeros([3])]
    return proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu
