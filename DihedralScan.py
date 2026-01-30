from ase.optimize.optimize import Optimizer
from ase import Atoms

class DihedralScan(Optimizer):


    def __init__(self, atoms: Atoms, dangle: float, total_steps: int, dihedral_idcs_list: list[int] | list[list[int]], mask_list: list[int] | list[list[int]] | None = None, **kwargs):
        self.dihedral_idcs_list = dihedral_idcs_list
        self.mask_list = mask_list
        self._check_argument_compatability(atoms, dihedral_idcs_list, mask_list)
        self.current_step = 0
        self.energy_list = []
        self.forces_list = []
        self.dihedral_list = []
        self.dangle = dangle
        self.total_steps = total_steps + 1
        super().__init__(atoms, **kwargs)

    def _check_argument_compatability(self, atoms, dihedral_idcs_list, mask_list):
        if not mask_list is None:
            if isinstance(mask_list[0], int):
                assert len(mask_list) == len(atoms), "Mask list length must match number of atoms."
            # elif isinstance(mask_list[0][0], int):
            else:
                for i, mask_sublist in enumerate(mask_list):
                    if not mask_sublist is None:
                        assert len(mask_sublist) == len(atoms), f"Each mask sublist length must match number of atoms. (entry {i} does not)"
        if isinstance(dihedral_idcs_list[0], int):
            assert len(dihedral_idcs_list) == 4, "Dihedral index list must have exactly 4 entries."
        elif isinstance(dihedral_idcs_list[0][0], int):
            for i, dihedral_idcs_sublist in enumerate(dihedral_idcs_list):
                assert len(dihedral_idcs_sublist) == 4, f"Each dihedral index sublist must have exactly 4 entries. (entry {i} does not)"
            if not mask_list is None:
                assert len(dihedral_idcs_list) == len(mask_list), "If multiple dihedral index sublists are provided, there must be a matching number of mask sublists or no masks at all."
            else:
                self.mask_list = [None] * len(dihedral_idcs_list)

    # def initialize(self):
    #     self.update(self.optimizable)
            
    def read(self):
        self.current_step, self.energy_list, self.forces_list, self.dihedral_list = self.load()

    def _step(self, optimizable):
        if isinstance(self.dihedral_idcs_list[0], int):
            dihedral_idcs = self.dihedral_idcs_list
            optimizable.atoms.rotate_dihedral(*dihedral_idcs, self.dangle, mask=self.mask_list)
        elif isinstance(self.dihedral_idcs_list[0][0], int):
            for i, dihedral_idcs in enumerate(self.dihedral_idcs_list):
                optimizable.atoms.rotate_dihedral(*dihedral_idcs, self.dangle, mask=self.mask_list[i])

    def _get_current_dihedral(self, optimizable):
        if isinstance(self.dihedral_idcs_list[0], int):
            dihedral_idcs = self.dihedral_idcs_list
            return optimizable.atoms.get_dihedral(*dihedral_idcs, mic=True)
        elif isinstance(self.dihedral_idcs_list[0][0], int):
            return optimizable.atoms.get_dihedral(*self.dihedral_idcs_list[0], mic=True)
        
    def update(self):
        forces = self.optimizable.atoms.get_forces()
        energy = self.optimizable.atoms.get_potential_energy()
        self.energy_list.append(energy)
        self.forces_list.append(forces)
        self.dihedral_list.append(self._get_current_dihedral(self.optimizable))
        self.dump((self.current_step, self.energy_list, self.forces_list, self.dihedral_list))
        print(f"Step {self.current_step}/{self.total_steps} ({self.dihedral_list[-1]:.2f}): Energy = {energy} eV")

    def step(self):
        optimizable = self.optimizable
        if not self.current_step == 0:
            self._step(optimizable)
        self.current_step += 1
        self.update()
        

    def converged(self, forces=None):
        conv = self.current_step >= self.total_steps
        if conv:
            print(f"Dihedral scan completed. (currently: {self.current_step}, target: {self.total_steps})")
        return conv