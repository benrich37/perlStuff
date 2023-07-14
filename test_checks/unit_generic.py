import unittest
from os.path import exists as ope
from os.path import join as opj
import os
import generic_helpers as gen
import subprocess
from ase import Atoms
from ase.constraints import FixBondLength

class TestGeomHelpers(unittest.TestCase):
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
        splitters = ["/", "\\", "\\\\"]
        for d in int_dirs:
            for s in splitters:
                if s in d:
                    self.assertIs(int(d.split(s)[-1]).__class__, int)
        gen.remove_dir_recursive(dir1)

    def test_log_generic(self):
        dir1 = 'tmp1'
        messages = ["message 1", "message2"]
        if ope(dir1):
            gen.remove_dir_recursive(dir1)
        os.mkdir(dir1)
        gen.log_generic(messages[0], dir1, "test", False)
        self.assertTrue(ope(opj(dir1, "test.log")))
        with open(opj(dir1, "test.log"), "r") as f:
            length = 0
            for i, line in enumerate(f):
                length += 1
                if i == 0:
                    self.assertTrue("Starting" in line)
                if i == 1:
                    self.assertTrue(messages[i - 1] in line)
            self.assertEqual(length, 2)
        gen.log_generic(messages[1], dir1, "test", False)
        with open(opj(dir1, "test.log"), "r") as f:
            length = 0
            for i, line in enumerate(f):
                length += 1
                if i == 0:
                    self.assertTrue("Starting" in line)
                if i >= 1:
                    self.assertTrue(messages[i - 1] in line)
            self.assertEqual(length, 3)
        gen.remove_dir_recursive(dir1)

