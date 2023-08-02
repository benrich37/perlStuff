import unittest
from os.path import exists as ope
from os.path import join as opj
import os
from helpers import generic_helpers as gen
from helpers import se_neb_helpers as se
from ase import Atoms
from ase.constraints import FixBondLength

class TestScheduleReaders(unittest.TestCase):
    def setUp(self):
        self.work_dir = opj(os.getcwd(), "tmp")
        if not ope(self.work_dir):
            os.mkdir(self.work_dir)
        self.step_atoms = [1, 2]
        self.scan_steps = 10
        self.step_size = 0.1
        self.guess_type = 1
        self.j_steps = 100
        self.freeze_list = [tuple([1, 2]), tuple([1, 2, 3])]
        self.custom_schedule = {
            "0": {
                "step_atoms": [2, 3],
                "step_size": 0.1,
                "guess_type": 1,
                "jdftx_steps": 100,
                "freeze_list": [[2, 3], [2, 3, 4]]
            },
            "1": {
                "step_atoms": [5, 3],
                "step_size": 0.3,
                "guess_type": 0,
                "jdftx_steps": 0,
                "freeze_list": [[5, 3], [2, 3, 4], [1, 2, 3]]
            },
            "2": {
                "step_atoms": [5, 10],
                "step_size": 0.3,
                "guess_type": 0,
                "jdftx_steps": 0,
                "freeze_list": [[5, 3], [2, 3, 4], [1, 2, 3]]
            }
        }

    def tearDown(self):
        if ope(self.work_dir):
            gen.remove_dir_recursive(self.work_dir)

    def test_self_consistency(self):
        se.write_auto_schedule(self.step_atoms, self.scan_steps, self.step_size, self.guess_type, self.j_steps,
                               self.freeze_list, self.work_dir)
        schedule = se.read_schedule_file(self.work_dir)
        # nSteps = len(list(schedule.keys))
        nSteps = len(schedule)
        self.assertEqual(nSteps,self.scan_steps)
        for i in range(nSteps):
            step_atoms = schedule[str(i)]["step_atoms"]
            for j, idx in enumerate(step_atoms):
                self.assertEqual(idx,self.step_atoms[j])
            self.assertEqual(self.guess_type,schedule[str(i)]["guess_type"])
            self.assertEqual(self.j_steps,schedule[str(i)]["jdftx_steps"])
            freeze_list = schedule[str(i)]["freeze_list"]
            for j, tup in enumerate(freeze_list):
                for k, idx in enumerate(tup):
                    self.assertEqual(self.freeze_list[j][k],idx)

    def test_custom_schedule(self):
        se.write_schedule_dict(self.custom_schedule, self.work_dir)
        new_sched = se.read_schedule_file(self.work_dir)
        for key1 in new_sched.keys():
            assert key1 in self.custom_schedule.keys()
            for key2 in new_sched[key1].keys():
                assert key2 in self.custom_schedule[key1].keys()
                if key2 in [se.step_size_key, se.j_steps_key, se.guess_type_key]:
                    self.assertEqual(new_sched[key1][key2], self.custom_schedule[key1][key2])
                elif key2 == se.step_atoms_key:
                    for i, idx in enumerate(new_sched[key1][key2]):
                        self.assertEqual(idx, self.custom_schedule[key1][key2][i])
                elif key2 == se.freeze_list_key:
                    for i, group in enumerate(new_sched[key1][key2]):
                        for j, idx in enumerate(group):
                            self.assertEqual(idx, self.custom_schedule[key1][key2][i][j])




