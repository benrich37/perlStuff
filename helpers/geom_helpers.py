import numpy as np
from ase.io import read
from scipy.spatial.transform import Rotation as R


def rotate_points(p1, p2, p3, target_angle):
    # Convert the target angle from degrees to radians
    target_angle_rad = np.radians(target_angle)

    # Define the vector from p2 to p1 and p2 to p3
    v21 = p1 - p2
    v23 = p3 - p2

    # Calculate the current angle between the vectors
    dot_product = np.dot(v21, v23)
    magnitude_product = np.linalg.norm(v21) * np.linalg.norm(v23)
    current_angle_rad = np.arccos(dot_product / magnitude_product)

    # Calculate the rotation axis using cross product
    rotation_axis = np.cross(v21, v23)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # normalize the vector

    # Calculate the required rotation angle
    rotation_angle_rad = (target_angle_rad - current_angle_rad)/2.

    # Perform rotation
    r1 = - rotation_angle_rad * rotation_axis
    r2 = rotation_angle_rad * rotation_axis

    # Rotate the points
    rotated_v21 = R.from_rotvec(r1).apply(v21)
    rotated_v23 = R.from_rotvec(r2).apply(v23)

    # Translate back to the original position
    rotated_p1 = rotated_v21 + p2
    rotated_p3 = rotated_v23 + p2

    return rotated_p1, rotated_p3


def get_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    theta = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return theta*(180/np.pi)


def get_bond_length(atoms, indices):
    posn1 = atoms.positions[indices[0]]
    posn2 = atoms.positions[indices[1]]
    return np.linalg.norm(posn2 - posn1)


def step_bond_with_momentum(atom_pair, step_length, atoms_prev_2, atoms_prev_1):
    target_length = get_bond_length(atoms_prev_1, atom_pair) + step_length
    dir_vecs = []
    for i in range(len(atoms_prev_1.positions)):
        dir_vecs.append(atoms_prev_1[i] - atoms_prev_2[i])
    for i in range(len(dir_vecs)):
        atoms_prev_1.positions[i] += dir_vecs[i]
    dir_vec = atoms_prev_1.positions[atom_pair[1]] - atoms_prev_1.positions[atom_pair[0]]
    cur_length = np.linalg.norm(dir_vec)
    should_be_0 = target_length - cur_length
    if not np.isclose(should_be_0, 0.0):
        atoms_prev_1.positions[atom_pair[1]] += dir_vec * should_be_0 / np.linalg.norm(dir_vec)
    return atoms_prev_1


def get_atoms_prep_follow(atoms, prev_2_out, atom_pair, target_length):
    atoms_prev = read(prev_2_out, format="vasp")
    dir_vecs = []
    for i in range(len(atoms.positions)):
        dir_vecs.append(atoms.positions[i] - atoms_prev.positions[i])
    for i in range(len(dir_vecs)):
        atoms.positions[i] += dir_vecs[i]
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    cur_length = np.linalg.norm(dir_vec)
    should_be_0 = target_length - cur_length
    if not np.isclose(should_be_0, 0.0):
        atoms.positions[atom_pair[1]] += dir_vec * (should_be_0) / np.linalg.norm(dir_vec)
    return atoms
