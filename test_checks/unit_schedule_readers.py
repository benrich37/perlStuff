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
        self.scan_steps = 20
        self.step_size = 0.1
        self.guess_type = 1
        self.j_steps = 100
        self.freeze_list = [tuple([1, 2]), tuple([1, 2, 3])]
        self.relax_start = True
        self.relax_end = True
        self.k = 0.1
        self.neb_method = "fake_method"
        self.neb_steps = 100
        self.custom_schedule = {
            "0": {
                se.step_atoms_key: [2, 3],
                se.step_size_key: 0.1,
                se.target_bool_key: False,
                se.guess_type_key: 1,
                se.j_steps_key: 100,
                se.freeze_list_key: [[2, 3], [2, 3, 4]]
            },
            "1": {
                se.step_atoms_key: [5, 3],
                se.step_size_key: 0.3,
                se.target_bool_key: False,
                se.guess_type_key: 0,
                se.j_steps_key: 0,
                se.freeze_list_key: [[5, 3], [2, 3, 4], [1, 2, 3]]
            },
            "2": {
                se.step_atoms_key: [5, 10],
                se.step_size_key: 0.3,
                se.target_bool_key: False,
                se.guess_type_key: 0,
                se.j_steps_key: 0,
                se.freeze_list_key: [[5, 3], [2, 3, 4], [1, 2, 3]]
            }
        }

    def tearDown(self):
        if ope(self.work_dir):
            gen.remove_dir_recursive(self.work_dir)

    def test_self_consistency(self):
        se.write_autofill_schedule(self.step_atoms, self.scan_steps, self.step_size, self.guess_type, self.j_steps,
                                   self.freeze_list, self.relax_start, self.relax_end, self.neb_steps, self.k,
                                   self.neb_method, self.work_dir)
        schedule = se.read_schedule_file(self.work_dir)
        nSteps = len(schedule) - 1
        self.assertEqual(nSteps,self.scan_steps)
        for i in range(nSteps):
            step_atoms = schedule[str(i)]["step_atoms"]
            for j, idx in enumerate(step_atoms):
                self.assertEqual(idx,self.step_atoms[j])
            self.assertEqual(self.guess_type,schedule[str(i)]["guess_type"])
            self.assertEqual(self.j_steps,schedule[str(i)]["jdftx_steps"])
            freeze_list = schedule[str(i)]["freeze_list"]
            if not (((i == 0) and self.relax_start) or ((i == self.scan_steps) and self.relax_end)):
                for j, tup in enumerate(freeze_list):
                    for k, idx in enumerate(tup):
                        self.assertEqual(self.freeze_list[j][k],idx)
            else:
                self.assertEqual(0, len(freeze_list))

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

    def test_key_ordering(self):
        se.write_autofill_schedule(self.step_atoms, self.scan_steps, self.step_size, self.guess_type, self.j_steps,
                                   self.freeze_list, self.relax_start, self.relax_end, self.neb_steps, self.k,
                                   self.neb_method, self.work_dir)
        schedule = se.read_schedule_file(self.work_dir)
        prev = None
        for key in schedule.keys():
            if prev is None:
                prev = int(key)
            else:
                if not key == se.neb_key:
                    self.assertTrue(int(key) > prev)




