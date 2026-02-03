from ase.optimize.optimize import Optimizer, OptimizableAtoms
from ase import Atoms
import numpy as np

def _rotate_substructure(atoms: OptimizableAtoms, axis_vector, center_vector, mol_idcs, dangle) -> None:
    work_atoms = atoms if not isinstance(atoms, OptimizableAtoms) else atoms.atoms
    axis_vector /= np.linalg.norm(axis_vector)
    # tmp_atoms: Atoms = work_atoms.copy()
    # for i in list(range(len(work_atoms)))[::-1]:
    #     if not i in mol_idcs:
    #         del tmp_atoms[i]
    tmp_atoms: Atoms = work_atoms[mol_idcs].copy()
    tmp_atoms.rotate(dangle, axis_vector, center=center_vector)
    posns = work_atoms.get_positions()
    for i, idx in enumerate(mol_idcs):
        posns[idx] = tmp_atoms[i].position
    work_atoms.set_positions(posns)

def rotate_substructure(atoms, mol_idcs, axis_idcs, center_idx, dangle) -> None:
    axis_vector = atoms[axis_idcs[1]].position - atoms[axis_idcs[0]].position
    axis_vector /= np.linalg.norm(axis_vector)
    center_vector = atoms[center_idx].position
    _rotate_substructure(atoms, axis_vector, center_vector, mol_idcs, dangle)

class RotationScan(Optimizer):

    ref_axes: dict[str, np.ndarray] = {
        "x": np.array([1., 0., 0.]),
        "y": np.array([0., 1., 0.]),
        "z": np.array([0., 0., 1.])
    }

    def __init__(self, atoms: Atoms, dangle: float, total_steps: int, 
                 mol_idcs: list[int] | None = None, 
                 axis: list[int] | str | np.ndarray | None = None,
                 center: int | list[int] | None = None,
                 init_step: float = 0.,
                 **kwargs):
        self.dangle = dangle
        self.total_steps = total_steps + 1
        self.init_step = init_step
        self.mol_idcs = self._resolve_mol_idcs(atoms, mol_idcs)
        self.axis = self._resolve_axis(atoms, axis)
        self.center = self._resolve_center(atoms, center)
        self.current_step = 0
        self.energy_list = []
        self.forces_list = []
        self.angle_list = []
        self.angle = 0.
        self.dangle = dangle
        
        super().__init__(atoms, **kwargs)

    def _resolve_mol_idcs(self, atoms, mol_idcs):
        if mol_idcs is None:
            return list(range(len(atoms)))
        return mol_idcs
    
    def _resolve_axis(self, atoms, axis):
        if axis is None:
            axis = "z"
        if isinstance(axis, str):
            if axis.lower() in self.ref_axes:
                return self.ref_axes[axis.lower()]
            else:
                raise ValueError(f"Invalid axis string '{axis}'. Must be one of {list(self.ref_axes.keys())}.")
        elif isinstance(axis, list):
            if not all(isinstance(i, int) for i in axis) or len(axis) != 2:
                raise ValueError(f"Axis list must contain exactly two integer indices. Got: {axis}")
            axis = atoms[axis[1]].position - atoms[axis[0]].position
        if isinstance(axis, np.ndarray):
            axis /= np.linalg.norm(axis)
            return axis
        raise ValueError(f"Invalid axis type: {type(axis)}. Must be str, list[int], or np.ndarray.")
    
    def _resolve_center(self, atoms, center):
        if center is None:
            center = np.mean(atoms[self.mol_idcs].positions, axis=0)
        elif isinstance(center, int):
            center = atoms[center].position
        elif isinstance(center, list):
            center = np.mean(atoms[center].positions, axis=0)
        return center
        

    def initialize(self):
        self._step(self.optimizable, self.init_step)
            
    def read(self):
        self.current_step, self.energy_list, self.forces_list, self.angle_list = self.load()
        # self.angle = self.angle_list[-1] if len(self.angle_list) > 0 else 0.
        self.angle = self.angle_list[-1]

    def _step(self, optimizable, dangle: float):
        _rotate_substructure(
            optimizable,
            self.axis,
            self.center,
            self.mol_idcs,
            dangle
        )
        self.angle += dangle
        
    def update(self):
        forces = self.optimizable.atoms.get_forces()
        energy = self.optimizable.atoms.get_potential_energy()
        self.energy_list.append(energy)
        self.forces_list.append(forces)
        self.angle_list.append(self.angle)
        self.dump((self.current_step, self.energy_list, self.forces_list, self.angle_list))
        print(f"Step {self.current_step}/{self.total_steps} ({self.angle:.2f}): Energy = {energy} eV")

    def step(self):
        optimizable = self.optimizable
        if not self.current_step == 0:
            self._step(optimizable, self.dangle)
        self.current_step += 1
        self.update()
        

    def converged(self, forces=None):
        conv = self.current_step >= self.total_steps
        if conv:
            print(f"Dihedral scan completed. (currently: {self.current_step}, target: {self.total_steps})")
        return conv
    