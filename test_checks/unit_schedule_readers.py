import unittest
from os.path import exists as ope
from os.path import join as opj
import os

import helpers.schedule_helpers
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
                helpers.schedule_helpers.step_atoms_key: [2, 3],
                helpers.schedule_helpers.step_size_key: 0.1,
                helpers.schedule_helpers.target_bool_key: False,
                helpers.schedule_helpers.guess_type_key: 1,
                helpers.schedule_helpers.j_steps_key: 100,
                helpers.schedule_helpers.freeze_list_key: [[2, 3], [2, 3, 4]],
                helpers.schedule_helpers.energy_key: None,
                helpers.schedule_helpers.properties_key: None,
            },
            "1": {
                helpers.schedule_helpers.step_atoms_key: [5, 3],
                helpers.schedule_helpers.step_size_key: 0.3,
                helpers.schedule_helpers.target_bool_key: False,
                helpers.schedule_helpers.guess_type_key: 0,
                helpers.schedule_helpers.j_steps_key: 0,
                helpers.schedule_helpers.freeze_list_key: [[5, 3], [2, 3, 4], [1, 2, 3]],
                helpers.schedule_helpers.energy_key: None,
                helpers.schedule_helpers.properties_key: None,
            },
            "2": {
                helpers.schedule_helpers.step_atoms_key: [5, 10],
                helpers.schedule_helpers.step_size_key: 0.3,
                helpers.schedule_helpers.target_bool_key: False,
                helpers.schedule_helpers.guess_type_key: 0,
                helpers.schedule_helpers.j_steps_key: 0,
                helpers.schedule_helpers.freeze_list_key: [[5, 3], [2, 3, 4], [1, 2, 3]],
                helpers.schedule_helpers.energy_key: None,
                helpers.schedule_helpers.properties_key: None,
            }
        }

    def tearDown(self):
        if ope(self.work_dir):
            gen.remove_dir_recursive(self.work_dir)

    def test_self_consistency(self):
        helpers.schedule_helpers.write_autofill_schedule(self.step_atoms, self.scan_steps, self.step_size, self.guess_type, self.j_steps,
                                                         self.freeze_list, self.relax_start, self.relax_end, self.neb_steps, self.k,
                                                         self.neb_method, self.work_dir)
        schedule = helpers.schedule_helpers.read_schedule_file(self.work_dir)
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
        helpers.schedule_helpers.write_schedule_dict(self.custom_schedule, self.work_dir)
        new_sched = helpers.schedule_helpers.read_schedule_file(self.work_dir)
        for key1 in new_sched.keys():
            assert key1 in self.custom_schedule.keys()
            for key2 in new_sched[key1].keys():
                assert key2 in self.custom_schedule[key1].keys()
                if key2 in [helpers.schedule_helpers.step_size_key, helpers.schedule_helpers.j_steps_key,
                            helpers.schedule_helpers.guess_type_key]:
                    self.assertEqual(new_sched[key1][key2], self.custom_schedule[key1][key2])
                elif key2 == helpers.schedule_helpers.step_atoms_key:
                    for i, idx in enumerate(new_sched[key1][key2]):
                        self.assertEqual(idx, self.custom_schedule[key1][key2][i])
                elif key2 == helpers.schedule_helpers.freeze_list_key:
                    for i, group in enumerate(new_sched[key1][key2]):
                        for j, idx in enumerate(group):
                            self.assertEqual(idx, self.custom_schedule[key1][key2][i][j])

    def test_key_ordering(self):
        helpers.schedule_helpers.write_autofill_schedule(self.step_atoms, self.scan_steps, self.step_size, self.guess_type, self.j_steps,
                                                         self.freeze_list, self.relax_start, self.relax_end, self.neb_steps, self.k,
                                                         self.neb_method, self.work_dir)
        schedule = helpers.schedule_helpers.read_schedule_file(self.work_dir)
        prev = None
        for key in schedule.keys():
            if prev is None:
                prev = int(key)
            else:
                if not key == helpers.schedule_helpers.neb_key:
                    self.assertTrue(int(key) > prev)

    def test_count_steps(self):
        expected = 3
        found = helpers.schedule_helpers.count_scan_steps_from_schedule(self.custom_schedule)
        self.assertEqual(expected, found)




