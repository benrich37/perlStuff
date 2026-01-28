from pyjdftx.ase import JDFTx as _JDFTx
from pyjdftx.ase._jdftx import reserved_commands, xc_map
from pymatgen.io.jdftx.inputs import JDFTXInfile
from ase.units import Hartree

smear_map = {
    "Fermi": "fermi-dirac",
    "Gaussian": "gaussian",
    "MP1": "methfessel-paxton",
    "Cold": "cold",
    }

def translate_infile_to_pydftx_kwargs(infile: JDFTXInfile, kwargs) -> dict:
    kwargs["kpts"] = tuple([int(infile["kpoint-folding"]["n0"]), 
                                int(infile["kpoint-folding"]["n1"]), 
                                int(infile["kpoint-folding"]["n2"])])
    if "elec-initial-charge" in infile:
        kwargs["initial_charges"] = -float(infile["elec-initial-charge"])
    kwargs["nbands"] = int(infile.get("elec-n-bands", {"n": 0})["n"])
    if "lattice-minimize" in infile:
        kwargs["variable_cell"] = infile["lattice-minimize"].get("nIterations", 0) > 0
    if "elec-smearing" in infile:
        smearing_type = smear_map[infile["elec-smearing"].get("type", "Fermi")]
        smearing_width = float(infile["elec-smearing"].get("width", 0.001)) * Hartree
        kwargs["smearing"] = (smearing_type, smearing_width)
    if "coulomb-truncation-embed" in infile:
        kwargs["center"] = (
            float(infile["coulomb-truncation-embed"]["c0"]),
            float(infile["coulomb-truncation-embed"]["c1"]),
            float(infile["coulomb-truncation-embed"]["c2"])
        )
    if "elec-ex-corr" in infile:
        xc_str = None
        xc_value = infile["elec-ex-corr"]
        if isinstance(xc_value, dict):
            xc_value = xc_value["funcXC"]
        for xc_nickname in xc_map:
            if xc_map[xc_nickname] == xc_value:
                xc_str = xc_nickname
                break
        if xc_str is None:
            raise ValueError(f"XC functional {xc_value} not recognized in pyjdftx.")
        kwargs["xc"] = xc_str
    return kwargs

def strip_infile_of_reserved_commands(infile: JDFTXInfile) -> JDFTXInfile:
    new_infile = infile.copy()
    if type(new_infile) in [JDFTXInfile, dict]:
        for cmd in reserved_commands:
            if cmd in new_infile:
                del new_infile[cmd]
    return new_infile
    

class JDFTx(_JDFTx):
    # PBC MUST BE ENCODED INTO THE ATOMS
    def __init__(self, 
                 infile: JDFTXInfile, 
                 pseudoSet='GBRV',
                 **kwargs):
        kwargs["pseudopotentials"] = pseudoSet
        kwargs = translate_infile_to_pydftx_kwargs(infile, kwargs)
        infile = strip_infile_of_reserved_commands(infile)
        kwargs["commands"] = str(infile)
        super().__init__(**kwargs)