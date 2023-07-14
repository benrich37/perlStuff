import unittest
from os.path import exists as ope
from os.path import join as opj
import os
import generic_helpers as gen
import subprocess

class TestGenericHelpers(unittest.TestCase):
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
        gen.log_generic("message", dir1, "test", False)
        self.assertTrue(ope(opj(dir1, "test.log")))
        with open(opj(dir1, "test.log"), "r") as f:

        gen.remove_dir_recursive(dir1)

