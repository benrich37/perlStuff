"""End-to-end tests that run opt4.py against a local (CPU) JDFTx build.

Each test makes a fresh calc directory with a periodic H2 molecule and runs
`python opt4.py` in it, the same way it is run on the cluster. JDFTx finishes
each run in a couple of seconds.

Configuration (environment variables):
    PERLSTUFF_TEST_JDFTX          CPU jdftx executable. Falls back to $JDFTx, then ../jdftx/build/jdftx.
    PERLSTUFF_TEST_JDFTX_PSEUDO   Pseudopotential root containing GBRV/. Falls back to the backup of the
                                  perlmutter pseudopotential dir (matches the remote layout), then
                                  $JDFTx_pseudo, then ../jdftx/build/pseudopotentials.
    PERLSTUFF_TEST_JDFTX_THREADS  Threads passed to jdftx with -c (default 4).
Tests are skipped if no working executable or pseudopotentials are found.

Run from the repo root:
    python -m unittest test_checks/local_opt4.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from os.path import exists as ope, join as opj
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

repo_dir = Path(__file__).resolve().parents[1]
opt4_script = repo_dir / "opt4.py"
sibling_jdftx_build = repo_dir.parent / "jdftx" / "build"
remote_like_pseudo_dir = "/Volumes/Ext4TB/scratch_backup/perl_home/pseudopotentials"

# Slightly stretched H2 (equilibrium is ~0.75 A) so the optimizers have something to do
h2_bond = 0.85
h2_atoms = Atoms("H2", positions=[[4., 4., 4.], [4. + h2_bond, 4.05, 4.]], cell=[8., 8., 8.], pbc=True)

# Note read_inputs_dict_helper drops single-word lines and any line containing "#" or "opt"
h2_inputs = [
    "elec-cutoff 15",
    "kpoint-folding 1 1 1",
    "electronic-scf nIterations 50",
    "core-overlap-check none",  # GBRV H core radii overlap at H2 bond lengths
]

max_steps = 3


def _find_jdftx():
    candidates = [os.environ.get("PERLSTUFF_TEST_JDFTX"), os.environ.get("JDFTx"), str(sibling_jdftx_build / "jdftx")]
    tried = []
    for exe in candidates:
        if not exe:
            continue
        try:
            subprocess.run([exe, "-h"], check=True, capture_output=True, timeout=30)
            return exe, None
        except Exception as e:
            tried.append(f"{exe} ({type(e).__name__})")
    return None, "No working CPU jdftx executable found, tried: " + ", ".join(tried)


def _find_pseudo_dir():
    candidates = [
        os.environ.get("PERLSTUFF_TEST_JDFTX_PSEUDO"), remote_like_pseudo_dir, os.environ.get("JDFTx_pseudo"),
        str(sibling_jdftx_build / "pseudopotentials"),
        ]
    for d in candidates:
        if d and ope(opj(d, "GBRV", "H.uspp")):
            return d, None
    return None, "No pseudopotential directory with GBRV/H.uspp found"


jdftx_exe, jdftx_skip_reason = _find_jdftx()
pseudo_dir, pseudo_skip_reason = _find_pseudo_dir()
n_threads = os.environ.get("PERLSTUFF_TEST_JDFTX_THREADS", "4")


class Opt4TestCase(unittest.TestCase):
    jdftx_cmd = None

    def setUp(self):
        # Resolved since opt4 logs paths via getcwd() (/var -> /private/var on macOS)
        self.work_dir = Path(tempfile.mkdtemp(prefix="opt4_test_")).resolve()
        # opt4.main() always prepends "srun" to the JDFTx command off of anvil, so shim it
        bin_dir = self.work_dir / "bin"
        bin_dir.mkdir()
        srun = bin_dir / "srun"
        srun.write_text('#!/bin/sh\nexec "$@"\n')
        srun.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = str(bin_dir) + os.pathsep + self.env["PATH"]
        self.env["JDFTx"] = self.jdftx_cmd
        self.env["JDFTx_GPU"] = self.jdftx_cmd
        if pseudo_dir is not None:
            self.env["JDFTx_pseudo"] = pseudo_dir
        self.ion_dir = self.work_dir / "ion_opt"
        self.lat_dir = self.work_dir / "lat_opt"

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def write_opt_input(self, **overrides):
        opt_input = {
            "structure": "POSCAR",
            "max_steps": str(max_steps),
            "gpu": "False",
            "restart": "True",
            "pbc": "True True True",
            "lattice steps": "0",
            "opt program": "jdft",
            "ddec6": "False",
        }
        opt_input.update(overrides)
        (self.work_dir / "opt_input").write_text("".join(f"{k}: {v}\n" for k, v in opt_input.items()))

    def write_calc_dir(self, **opt_input_overrides):
        h2_atoms.write(self.work_dir / "POSCAR", format="vasp")
        (self.work_dir / "inputs").write_text("\n".join(h2_inputs) + "\n")
        self.write_opt_input(**opt_input_overrides)

    def run_opt4(self):
        return subprocess.run(
            [sys.executable, str(opt4_script)], cwd=self.work_dir, env=self.env,
            capture_output=True, text=True, timeout=600,
            )

    def assert_ran(self, proc):
        self.assertEqual(proc.returncode, 0, msg=f"opt4.py failed:\n{proc.stdout}\n{proc.stderr}\n{self.iolog()}")

    def iolog(self):
        fname = self.work_dir / "opt.iolog"
        return fname.read_text() if fname.exists() else ""

    def run_text(self, fname, calc_dir=None):
        calc_dir = self.ion_dir if calc_dir is None else calc_dir
        return (calc_dir / "jdftx_run" / fname).read_text()


@unittest.skipIf(pseudo_dir is None, pseudo_skip_reason)
class TestOpt4DryRun(Opt4TestCase):
    # These abort before JDFTx is called, so any executable path works
    jdftx_cmd = "false"

    def test_missing_opt_input_dumps_template(self):
        proc = self.run_opt4()
        self.assertEqual(proc.returncode, 1)
        self.assertIn("structure: POSCAR", (self.work_dir / "opt_input").read_text())

    def test_missing_inputs_dumps_defaults(self):
        h2_atoms.write(self.work_dir / "POSCAR", format="vasp")
        self.write_opt_input()
        proc = self.run_opt4()
        self.assertEqual(proc.returncode, 1)
        self.assertTrue((self.work_dir / "inputs").exists())
        self.assertFalse((self.ion_dir / "jdftx_run" / "in").exists())


@unittest.skipIf(jdftx_exe is None, jdftx_skip_reason)
@unittest.skipIf(pseudo_dir is None, pseudo_skip_reason)
class TestOpt4LocalJDFTx(Opt4TestCase):
    jdftx_cmd = f"{jdftx_exe} -c {n_threads}"

    def test_native_ionic_opt(self):
        self.write_calc_dir()
        self.assert_ran(self.run_opt4())
        self.assertTrue((self.ion_dir / "finished.txt").exists())
        infile = self.run_text("in")
        self.assertIn(f"nIterations  {max_steps}", infile)
        for dump_var in ["ElecDensity", "BandProjections", "State"]:
            self.assertIn(dump_var, infile)
        self.assertIn("band-projection-params yes no", infile)
        self.assertIn("ion-species " + opj(pseudo_dir, "GBRV", "H.uspp"), infile)
        out = self.run_text("out")
        self.assertTrue(out.rstrip().endswith("Done!"))
        self.assertIn(f"IonicMinimize: Iter:   {max_steps}", out)
        self.assertTrue((self.ion_dir / "jdftx_run" / "n").exists())
        relaxed = read(self.ion_dir / "CONTCAR", format="vasp")
        self.assertLess(relaxed.get_distance(0, 1), h2_bond - 0.02)

    def test_rerun_after_finish_skips_jdftx(self):
        self.write_calc_dir()
        self.assert_ran(self.run_opt4())
        out_before = self.run_text("out")
        self.assert_ran(self.run_opt4())
        self.assertEqual(self.run_text("out"), out_before)

    def test_restart_from_unfinished_ion_opt(self):
        self.write_calc_dir()
        self.assert_ran(self.run_opt4())
        first_bond = read(self.ion_dir / "CONTCAR", format="vasp").get_distance(0, 1)
        (self.ion_dir / "finished.txt").unlink()
        self.assert_ran(self.run_opt4())
        self.assertIn(f"Atoms object set from {self.ion_dir / 'jdftx_run' / 'out'}", self.iolog())
        self.assertTrue((self.ion_dir / "finished.txt").exists())
        # jdftx appends to the existing out file
        out = self.run_text("out")
        self.assertEqual(out.count("Done!"), 2)
        # Second run starts from the relaxed geometry, not the POSCAR
        self.assertLess(first_bond, h2_bond - 0.02)
        second_bond = read(self.ion_dir / "CONTCAR", format="vasp").get_distance(0, 1)
        self.assertLessEqual(abs(second_bond - first_bond), abs(h2_bond - first_bond))

    def test_ase_ionic_opt(self):
        self.write_calc_dir(**{"opt program": "ase"})
        self.assert_ran(self.run_opt4())
        self.assertTrue((self.ion_dir / "finished.txt").exists())
        self.assertIn("nIterations  0", self.run_text("in"))
        self.assertTrue((self.ion_dir / "opt.logx").exists())
        traj = read(self.ion_dir / "opt.traj", index=":")
        self.assertEqual(len(traj), max_steps + 1)
        self.assertLess(traj[-1].get_potential_energy(), traj[0].get_potential_energy())

    def test_no_restart(self):
        self.write_calc_dir(restart="False")
        self.assert_ran(self.run_opt4())
        self.assertTrue((self.ion_dir / "finished.txt").exists())

    def test_lattice_opt(self):
        self.write_calc_dir(**{"lattice steps": "2"})
        self.assert_ran(self.run_opt4())
        self.assertTrue((self.lat_dir / "finished.txt").exists())
        self.assertIn("LatticeMinimize: Iter:   2", self.run_text("out", self.lat_dir))
        self.assertTrue((self.ion_dir / "finished.txt").exists())
        # Ionic opt starts from the relaxed lattice
        lat_cell = read(self.lat_dir / "CONTCAR", format="vasp").cell
        self.assertGreater(abs(lat_cell.lengths() - h2_atoms.cell.lengths()).max(), 1e-4)
        ion_cell = read(self.ion_dir / "CONTCAR", format="vasp").cell
        self.assertTrue(np.allclose(ion_cell, lat_cell, atol=1e-5), msg=f"{ion_cell} != {lat_cell}")


if __name__ == "__main__":
    unittest.main()
