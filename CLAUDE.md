# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python scripts that drive JDFTx DFT calculations through ASE on HPC clusters (NERSC Perlmutter, also Anvil). There is no package, build step, or linter. Each top-level script (`opt*.py`, `scan_*.py`, `neb*.py`, `vib*.py`, `bias_scan.py`, ...) is a standalone entry point. You run it from inside a *calculation directory*, not from the repo: `python $pstuff/opt4.py`, where `$pstuff` points at this clone. Python puts the script's own directory on `sys.path`, which is how `helpers.*`, `scripts.*` and `JDFTx*.py` get imported.

## Tests

Plain `unittest` with no runner config. Run from the repo root, because tests import `helpers` and create/delete `tmp*` dirs in the cwd:

```
python -m unittest test_checks/unit_helpers.py
python -m unittest test_checks/unit_schedule_readers.py
python -m unittest test_checks.unit_helpers.TestGeomHelpers.test_read_pbc   # single test
```

[test_checks/local_opt4.py](test_checks/local_opt4.py) runs `opt4.py` end to end against a local CPU JDFTx build, using a tiny H2 system (~16 s total). It shims `srun` and skips if no working `jdftx` is found. The executable comes from `PERLSTUFF_TEST_JDFTX`, then `$JDFTx`, then `../jdftx/build/jdftx`. Pseudopotentials default to `/Volumes/Ext4TB/scratch_backup/perl_home/pseudopotentials`, which matches the remote layout.
```
PERLSTUFF_TEST_JDFTX=/path/to/jdftx python -m unittest -v test_checks/local_opt4.py
```

Development uses the conda env `research`. `pymatgen.io.jdftx` (`JDFTXInfile`, `JDFTXOutfile`) comes from an editable checkout of pymatgen-core at `../pymatgen-core`, not from a released pymatgen. Changes or uncommitted work there directly change the `in` files these scripts write.

## Run workflow (from README)

The scripts are designed to be run "dry" on a login node first. Each dry run exits early:
1. If `<script>_input` is missing (e.g. `opt_input`), a template is dumped and the script aborts. This file holds script settings as `key: value # comment` lines, parsed by `get_inputs_list`, and keys are matched loosely by substring.
2. If `psubmit.sh` is missing, a slurm template is dumped (`check_submit`). Newer scripts such as `opt4.py` have this call commented out.
3. After that, submit with `sbatch psubmit.sh`. Progress goes to `<calc_type>.iolog` (via `get_log_fn`). Optimizations are also written as `.logx` files (Gaussian-log format readable by GaussView). These store Lowdin charges labeled as Mulliken.

`debug = "perlStuff" in cwd`: running a script with the cwd inside this repo flips it into debug mode (no `srun`).

## JDFTx command flow

- The calc directory's `inputs` file holds JDFTx commands. If it is missing, an existing `in` is used; if neither exists, default inputs are dumped and the script aborts. Handled in `get_cmds_dict` / `read_inputs_dict_helper` in [helpers/generic_helpers.py](helpers/generic_helpers.py).
- `read_inputs_dict_helper` drops an `ignore` list of tags that the scripts manage themselves (ion-species, coords-type, lattice-type, ...). It joins `\` line continuations and merges repeated `add-U` lines. A `*` value for `elec-n-bands`, `kpoint-folding` or `target-mu` is computed from the structure, pseudopotential valence counts, or the `bias` setting.
- The resulting cmds dict/list is extended (`add_cohp_cmds`, `add_elec_density_dump`), then converted with `cmds_list_to_infile` into a pymatgen `JDFTXInfile`.
- Calculator generations (all ASE calculators, built in [helpers/calc_helpers.py](helpers/calc_helpers.py)):
  - [JDFTx.py](JDFTx.py): legacy, takes a command list (`_get_calc`).
  - [JDFTx_new.py](JDFTx_new.py): takes a `JDFTXInfile` and shells out to the executable (`_get_calc_new`). `opt4.py` uses this one.
  - [JDFTx_pyjdftx.py](JDFTx_pyjdftx.py): runs JDFTx in-process via `pyjdftx` + `mpi4py`. It translates the infile to pyjdftx kwargs and strips pyjdftx-reserved commands. `opt6*.py` use this one.
- Executable: `get_exe_cmd` reads `$JDFTx` or `$JDFTx_GPU` and prepends `srun ` unless in debug mode or the work dir path contains `anvil`.

Environment variables used: `JDFTx`, `JDFTx_GPU`, `JDFTx_pseudo` (pseudopotential root, with `pseudoset` as subdir), `DDEC6_EXE_PATH`, `DDEC6_AD_PATH`, `SLURM_NNODES`.

## Optimization script layout (opt4.py as reference)

The script works in the calc dir. It runs an optional lattice optimization in `lat_opt/`, then an ionic optimization in `ion_opt/`, using either the JDFTx internal optimizer or ASE `FIRE` (`opt program`). A finished stage is marked by `finished.txt`. On `restart`, structures and state files (`wfns`, `eigenvals`, `fillings`, `fluidState`) are recovered from those dirs (`copy_best_state_files`). The electron density is always dumped. The `ddec6` flag only controls whether [scripts/run_ddec6_v3.py](scripts/run_ddec6_v3.py) runs afterwards. Freezing atoms (`freeze base/tol/count`, `exclude_freeze_count`, `freeze_map`, `freeze_all_but_map`) goes through `get_apply_freeze_func`.

## Versioned and duplicated code: keep in sync

- Numbered scripts (`opt.py` … `opt6.py`, `neb_custom_images*.py`, `run_ddec6*.py`, `vib.py`/`vib2.py`) are successive generations kept side by side, not dead code. Check git log to see which is in active use. Recently that has been `opt4.py`, `opt6_self_contained.py` and `ase_opt_self_contained/`.
- `opt6_self_contained.py`, `opt6_self_contained copy.py`, `ase_opt_self_contained/opt.py` and [helpers/generic_helpers_nopmg.py](helpers/generic_helpers_nopmg.py) inline copies of the `generic_helpers` functions. That way they run on machines without the full repo (e.g. Anvil). `generic_helpers_nopmg.py` and `ase_opt_self_contained/JDFTx_pyjdftx.py` also drop the pymatgen dependency, but the self-contained `opt` scripts still import pymatgen. A fix to a helper such as `read_inputs_dict_helper` usually has to be applied to all of these copies as well as `helpers/generic_helpers.py`.
- Older scripts (`neb*.py`, `scan_bond_new.py`, `se_neb_fly.py`, ...) hard-code Windows `JDFTx_pseudo` paths in their debug branches.
