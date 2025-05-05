from os import getcwd as getcwd, environ as env_vars_dict
from JDFTx import JDFTx, Wannier
from JDFTx_new import JDFTx as JDFTx_new, JDFTXInfile
from helpers.generic_helpers import log_def, fix_dump_cmds_list


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
