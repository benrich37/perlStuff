from os import getcwd as getcwd, environ as env_vars_dict
from JDFTx import JDFTx
from helpers.generic_helpers import log_def


def set_calc(exe_cmd, cmds, work=getcwd(), debug=False, debug_calc=None):
    if debug:
        return debug_calc()
    else:
        return JDFTx(
            executable=exe_cmd,
            pseudoSet="GBRV_v1.5",
            commands=cmds,
            outfile=work,
            ionic_steps=False,
            ignoreStress=True,
    )


def _get_calc_old(exe_cmd, cmds, root, jdftx_fn, debug=False, debug_fn=None, log_fn=log_def):
    print("Ben - use set_calc instead")
    if debug:
        log_fn("Setting calc to debug calc")
        return debug_fn()
    else:
        log_fn(f"Setting calculator with \n \t exe_cmd: {exe_cmd} \n \t calc dir: {root} \n \t cmds: {cmds} \n")
        return jdftx_fn(
            executable=exe_cmd,
            pseudoSet="GBRV_v1.5",
            commands=cmds,
            outfile=root,
            ionic_steps=False
        )


def get_exe_cmd(gpu, log_fn):
    if gpu:
        _get = 'JDFTx_GPU'
    else:
        _get = 'JDFTx'
    log_fn(f"Using {_get} for JDFTx exe")
    exe_cmd = 'srun ' + env_vars_dict[_get]
    log_fn(f"exe_cmd: {exe_cmd}")
    return exe_cmd
