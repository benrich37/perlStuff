from os import getcwd as getcwd, environ as env_vars_dict
from os.path import exists as ope
from shutil import which
from JDFTx import JDFTx, Wannier
from JDFTx_new import JDFTx as JDFTx_new, JDFTXInfile
from helpers.generic_helpers import log_def, fix_dump_cmds_list, log_and_abort
from pathlib import Path


def set_calc_old(exe_cmd, cmds, work=getcwd(), debug=False, debug_calc=None):
    print("Ben - stop using this one it's redundant")
    if debug:
        return debug_calc()
    else:
        return JDFTx(
            executable=exe_cmd,
            pseudoSet="GBRV_v1.5",
            commands=cmds,
            calc_dir=work,
            ionic_steps=False,
            ignoreStress=True,
    )


def _get_wannier_calc(exe_cmd, cmds, root, pseudoSet="GBRV", gpu=False, log_fn=log_def):
    cmds = fix_dump_cmds_list(cmds)
    log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {root} \n \t cmds: {cmds} \n")
    return Wannier(
        executable=exe_cmd,
        pseudoSet=pseudoSet,
        commands=cmds,
        outfile=root,
        gpu=gpu
    )

def _get_calc(exe_cmd, cmds, root, pseudoSet="GBRV", debug=False, debug_fn=None, log_fn=log_def, direct_coords=False):
    cmds = fix_dump_cmds_list(cmds)
    if debug:
        log_fn("Setting calc to debug calc")
        return debug_fn()
    else:
        log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {root} \n \t cmds: {cmds} \n")
        return JDFTx(
            executable=exe_cmd,
            pseudoSet=pseudoSet,
            commands=cmds,
            calc_dir=root,
            ionic_steps=False,
            direct_coords=direct_coords
        )
    
def _get_calc_new(
        exe_cmd, cmds: list[str], root, pseudoSet="GBRV", pseudoDir=None, 
        debug=False, debug_fn=None, log_fn=log_def, direct_coords=False, label=None,
        ignore_cache_for_aimd=True, use_cart=False,
        ):
    cmds = fix_dump_cmds_list(cmds)
    if label is None:
        label = root
    else:
        label = str(Path(root) / label)
    force_eval = False
    if ignore_cache_for_aimd:
        if not isinstance(cmds, JDFTXInfile):
            log_fn("Cannot check if AIMD is requested since cmds is not a JDFTXInfile instance")
        elif "ionic-dynamics" in cmds and int(cmds["ionic-dynamics"]["nSteps"]) > 0:
            log_fn("AIMD requested - forcing evaluation even if atoms are unchanged")
            force_eval = True
        else:
            log_fn("AIMD not requested - using caching if possible")

    # log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {root} \n \t cmds: {cmds} \n")
    # infile = JDFTXInfile.from_str("" + "\n".join(list), dont_require_structure=True)
    return JDFTx_new(
        infile=cmds,
        label=label,
        pseudoDir=pseudoDir,
        pseudoSet=pseudoSet,
        command=exe_cmd,
        #debug=debug,
        log_func=log_fn,
        force_evaluation=force_eval,
        use_cart=use_cart,
    )

def backup_pyjdftx_out_file(outfile: Path, log_fn=log_def):
    outfile_name = outfile.name
    cur_fs = outfile.parent.iterdir()
    cur_backups = [f for f in cur_fs if f.is_file() and f.name.startswith(outfile_name)]
    cur_backups_end_parts = [f.name.split(outfile_name)[-1] for f in cur_backups]
    cur_backups_idcs = []
    for part in cur_backups_end_parts:
        idx = -1
        try:
            idx = int(part)
        except ValueError:
            pass
        cur_backups_idcs.append(idx)
    max_idx = max(cur_backups_idcs) if len(cur_backups_idcs) > 0 else -1
    backup_name = f"{outfile_name}{max_idx + 1}"
    backup_path = outfile.parent / backup_name
    outfile.rename(backup_path)
    log_fn(f"Backed up existing output file {outfile} to {backup_path}")

def get_calc_pyjdftx(
        cmds: list[str], root, pseudoSet="GBRV", pseudoDir=None, 
        debug=False, debug_fn=None, log_fn=log_def, direct_coords=False, label=None,
        ignore_cache_for_aimd=True,
        restart=True,
        ):
    from JDFTx_pyjdftx import JDFTx
    import pyjdftx
    from mpi4py import MPI
    
    cmds = fix_dump_cmds_list(cmds)
    if label is None:
        label = "jdftx"
    outfile = Path(root) / f"{label}.out"
    append = False
    if outfile.exists():
        log_fn(f"Output file {outfile} already exists. Backing up old output file.")
        backup_pyjdftx_out_file(outfile)
        # append = True
    pyjdftx.initialize(MPI.COMM_WORLD, MPI.COMM_WORLD, str(outfile), append)
    force_eval = False
    if ignore_cache_for_aimd:
        if not isinstance(cmds, JDFTXInfile):
            log_fn("Cannot check if AIMD is requested since cmds is not a JDFTXInfile instance")
        elif "ionic-dynamics" in cmds and int(cmds["ionic-dynamics"]["nSteps"]) > 0:
            log_fn("AIMD requested - forcing evaluation even if atoms are unchanged")
            force_eval = True
        else:
            log_fn("AIMD not requested - using caching if possible")

    # log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {root} \n \t cmds: {cmds} \n")
    # infile = JDFTXInfile.from_str("" + "\n".join(list), dont_require_structure=True)
    try:
        return JDFTx(
            directory=str(root),
            infile=cmds,
            label=label,
            pseudoSet=pseudoSet,
            restart=str(Path(root) / f"{label}") if restart else None,
    )
    except FileNotFoundError as e:
        log_fn(f"Could not find restart files at {str(Path(root) / f'{label}')}, starting fresh calculation")
        return JDFTx(
            directory=str(root),
            infile=cmds,
            label=label,
            pseudoSet=pseudoSet,
        ) 
    
# def _get_calc_new(
#         exe_cmd: str, infile: JDFTXInfile, calc_dir: str, pseudoSet="GBRV", debug=False, debug_fn=None, log_fn=log_def, direct_coords=False
#         ):
#     cmds = fix_dump_cmds_list(cmds)
#     if debug:
#         log_fn("Setting calc to debug calc")
#         return debug_fn()
#     else:
#         log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {calc_dir} \n \t cmds: {cmds} \n")
#         return JDFTx_new(

#             executable=exe_cmd,
#             pseudoSet=pseudoSet,
#             commands=cmds,
#             calc_dir=calc_dir,
#             ionic_steps=False,
#             direct_coords=direct_coords
#         )


def get_wannier_exe_cmd(gpu, log_fn):
    if gpu:
        _get = 'Wannier_GPU'
    else:
        _get = 'Wannier'
    log_fn(f"Using {_get} for Wannier exe")
    exe_cmd = 'srun ' + env_vars_dict[_get]
    log_fn(f"exe_cmd: {exe_cmd}")
    return exe_cmd

local_env_key = "PERLSTUFF_LOCAL"


def is_local_run():
    """Return True if running on a local machine rather than a slurm cluster.

    Detected by the absence of slurm (no srun on PATH and no SLURM_JOB_ID).
    Set $PERLSTUFF_LOCAL to 1 or 0 to override the detection.
    """
    override = env_vars_dict.get(local_env_key, "").strip().lower()
    if override:
        return override in ["1", "true", "yes"]
    return (which("srun") is None) and ("SLURM_JOB_ID" not in env_vars_dict)


def apply_local_protections(oid: dict, log_fn=log_def) -> dict:
    """Override opt_input settings that only make sense on remote machines when running locally.

    Sets oid["local"]. On a local run, uses the CPU executable instead of the GPU one, and only
    keeps DDEC6 if a chargemol executable is available. Callers should also skip srun if oid["local"].
    """
    oid["local"] = is_local_run()
    if not oid["local"]:
        return oid
    log_fn(f"Local run detected (no slurm found, or ${local_env_key} set) - overriding remote-only settings")
    if oid.get("gpu"):
        log_fn("Local run: ignoring 'gpu: True' - using the CPU executable ($JDFTx)")
        oid["gpu"] = False
    if oid.get("ddec6") and not ope(env_vars_dict.get("DDEC6_EXE_PATH", "")):
        log_fn("Local run: $DDEC6_EXE_PATH is not an existing chargemol executable - skipping DDEC6 analysis")
        oid["ddec6"] = False
    if not "JDFTx" in env_vars_dict:
        log_and_abort("Local run: $JDFTx must be set to a CPU JDFTx executable", log_fn=log_fn)
    return oid


def get_exe_cmd(gpu, log_fn, use_srun=True):
    if gpu:
        _get = 'JDFTx_GPU'
    else:
        _get = 'JDFTx'
    log_fn(f"Using {_get} for JDFTx exe")
    if use_srun:
        exe_cmd = 'srun ' + env_vars_dict[_get]
    else:
        exe_cmd = env_vars_dict[_get]
    log_fn(f"exe_cmd: {exe_cmd}")
    return exe_cmd

def get_nNodes(log_fn):
    key = "SLURM_NNODES"
    if key in env_vars_dict:
        nNodes = int(env_vars_dict[key])
        log_fn(f"Found {nNodes} nodes available")
        return nNodes
    else:
        nNodes = 1
        log_fn(f"Could not find number of nodes ($SLURM_NNODES missing from environmental variables). Setting number of nodes to 1")
        return nNodes

def get_exe_cmd_test(gpu, log_fn):
    if gpu:
        _get = 'JDFTx_GPU'
    else:
        _get = 'JDFTx'
    log_fn(f"Using {_get} for JDFTx exe")
    nNodes = get_nNodes(log_fn)
    exe_cmd = f'srun -N {nNodes} ' + env_vars_dict[_get]
    log_fn(f"exe_cmd: {exe_cmd}")
    return exe_cmd
