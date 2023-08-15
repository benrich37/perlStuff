import unittest
from os.path import exists as ope, join as opj, basename as basename
import os
from helpers import generic_helpers as gen
from ase import Atoms
from ase.constraints import FixBondLength
from helpers.neb_helpers import get_good_idcs_helper

class TestGeomHelpers(unittest.TestCase):
    def test_read_pbc(self):
        line1 = "pbc: True true false"
        val1 = line1.rstrip("\n").split(":")[1]
        pbc1 = gen.read_pbc_val(val1)
        self.assertTrue(pbc1[0])
        self.assertTrue(pbc1[1])
        self.assertFalse(pbc1[2])
        line2 = "pbc: False false false"
        val2 = line2.rstrip("\n").split(":")[1]
        pbc2 = gen.read_pbc_val(val2)
        self.assertFalse(pbc2[0])
        self.assertFalse(pbc2[1])
        self.assertFalse(pbc2[2])
        line3 = "pbc: True, true, true"
        val3 = line3.rstrip("\n").split(":")[1]
        pbc3 = gen.read_pbc_val(val3)
        self.assertTrue(pbc3[0])
        self.assertTrue(pbc3[1])
        self.assertTrue(pbc3[2])

    def test_add_constraint(self):
        atoms = Atoms('HOH',
                      positions=[[0, 0, -1], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(len(atoms.constraints), 0)
        gen.add_constraint(atoms, FixBondLength(0, 1))
        self.assertEqual(len(atoms.constraints), 1)
        gen.add_constraint(atoms, FixBondLength(0, 2))
        self.assertEqual(len(atoms.constraints), 2)

    def test_add_bond_constraints(self):
        atoms = Atoms('HOH', positions=[[0, 0, -1], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(len(atoms.constraints), 0)
        self.assertRaises(ValueError, lambda: gen.add_bond_constraints(atoms, [0, 1, 2]))
        gen.add_bond_constraints(atoms, [0, 1, 0, 2])
        self.assertEqual(len(atoms.constraints), 2)
        atoms = Atoms('HOH', positions=[[0, 0, -1], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(len(atoms.constraints), 0)
        gen.add_bond_constraints(atoms, [0, 1])
        self.assertEqual(len(atoms.constraints), 1)

    def test_sort_bool(self):
        test1 = ["H", "H", "O", "H"]
        test1_bool = gen.get_sort_bool(test1)
        self.assertTrue(test1_bool)
        test2 = ["H", "H", "H", "O"]
        test2_bool = gen.get_sort_bool(test2)
        self.assertFalse(test2_bool)


class TestIoHelpers(unittest.TestCase):
    def test_copy_files(self):
        dir1 = './tmp1'
        dir2 = './tmp2'
        for d in [dir1, dir2]:
            if ope(d):
                gen.remove_dir_recursive(d)
        os.mkdir(dir1)
        os.mkdir(dir2)
        fnames = ['ex1.txt', 'ex2.txt']
        for f in fnames:
            with open(opj(dir1, f), "w") as file:
                file.write("exists")
        gen.copy_files(dir1, dir2)
        for f in fnames:
            self.assertTrue(ope(opj(dir2, f)))
        for d in [dir1, dir2]:
            gen.remove_dir_recursive(d)

    def test_get_int_dirs(self):
        dir1 = 'tmp1'
        if ope(dir1):
            gen.remove_dir_recursive(dir1)
        os.mkdir(dir1)
        nIntDirs = 20
        for i in range(nIntDirs):
            os.mkdir(opj(dir1, str(i)))
        int_dirs = gen.get_int_dirs(dir1)
        self.assertEqual(len(int_dirs), nIntDirs)
        for d in int_dirs:
            self.assertEqual(int_dirs.count(d), 1)
        last_int = None
        for d in int_dirs:
            int_name = int(basename(d))
            self.assertIs(int_name.__class__, int)
            if last_int is None:
                last_int = int_name
            else:
                self.assertTrue(int_name > last_int)
        gen.remove_dir_recursive(dir1)

    def test_log_generic(self):
        dir1 = 'tmp1'
        messages = ["message 1", "message2"]
        if ope(dir1):
            gen.remove_dir_recursive(dir1)
        os.mkdir(dir1)
        fname="test"
        gen.log_generic(messages[0], dir1, fname, False)
        self.assertTrue(ope(opj(dir1, fname)))
        with open(opj(dir1, fname), "r") as f:
            length = 0
            for i, line in enumerate(f):
                length += 1
                if i == 0:
                    self.assertTrue("Starting" in line)
                if i == 1:
                    self.assertTrue(messages[i - 1] in line)
            self.assertEqual(length, 2)
        gen.log_generic(messages[1], dir1, fname, False)
        with open(opj(dir1, fname), "r") as f:
            length = 0
            for i, line in enumerate(f):
                length += 1
                if i == 0:
                    self.assertTrue("Starting" in line)
                if i >= 1:
                    self.assertTrue(messages[i - 1] in line)
            self.assertEqual(length, 3)
        gen.remove_dir_recursive(dir1)


class TestNebHelper(unittest.TestCase):
    def test_get_good_idcs(self):
        test_idcs_1 = [0, 1, 2, 3, 2, 2, 2.5, 1]
        test_idcs_1_gidcs = [0, 1, 2, 3, 4, 5, 7]
        test_idcs_2 = [5, 6, 1, 2, 7, 5, 4]
        test_idcs_2_gidcs = [2, 3, 4, 5, 6]
        test_ins = [test_idcs_1, test_idcs_2]
        test_outs = [test_idcs_1_gidcs, test_idcs_2_gidcs]
        for i in range(len(test_ins)):
            output = get_good_idcs_helper(test_ins[i])
            self.assertEqual(len(output), len(test_outs[i]))
            for j in range(len(output)):
                self.assertEqual(output[j], test_outs[i][j])