#!/usr/bin/env python

# Atomistic Simulation Environment (ASE) calculator interface for JDFTx
# See http://jdftx.org for JDFTx and https://wiki.fysik.dtu.dk/ase/ for ASE
# Authors: Deniz Gunceler, Ravishankar Sundararaman

from __future__ import print_function #For Python2 compatibility

import numpy as np
import scipy, subprocess, re
from ase.calculators.calculator import Calculator
# from ase.calculators.interface import Calculator
from ase.units import Bohr, Hartree
from ase import Atoms
from os import environ as env_vars_dict
from os.path import join as opj
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pathlib import Path


def debugging_override():
        return False
        # return env_vars_dict["debugging_jdftx_for_perl"] == "True"




#Run shell command and return output as a string:
def shell(cmd):
        if not debugging_override():
                return subprocess.check_output(cmd, shell=True)

#Return var, replacing it with environment variable varName if it is None
def replaceVariable(var, varName):
        if (var is None) and (varName in env_vars_dict):
                return env_vars_dict[varName]
        else:
                return var

class JDFTx(Calculator):

        implemented_properties = ['energy', 'forces']

        def __init__(self, executable=None, pseudoDir=None, pseudoSet='GBRV', commands=None, calc_dir=None,
                     ionic_steps = False, ignoreStress=True, direct_coords=False):
                self.atoms = None  # copy of atoms object from last calculation
                self.results = {}  # calculated properties (energy, forces, ...)
                self.parameters = {}  # calculational parameters
                super().__init__(label=calc_dir)
                #self._directory = None  # Initialize
                #Valid pseudopotential sets (mapping to path and suffix):
                pseudoSetMap = {
                        'SG15' : 'SG15/$ID_ONCV_PBE.upf',
                        'GBRV' : 'GBRV/$ID_pbe.uspp',
        				'GBRV_v1.5' : 'GBRV_v1.5/$ID_pbe_v1.uspp',
                        'GBRV-pbe' : 'GBRV/$ID_pbe.uspp',
                        'GBRV-lda' : 'GBRV/$ID_lda.uspp',
                        'GBRV-pbesol' : 'GBRV/$ID_pbesol.uspp',
                        'kjpaw': 'kjpaw/$ID_pbe-n-kjpaw_psl.1.0.0.upf',
                        'dojo': 'dojo/$ID.upf',
                }
                self.direct = direct_coords
                #Get default values from environment:
                self.executable = replaceVariable(executable, 'JDFTx')      #Path to the jdftx executable (cpu or gpu)
                self.pseudoDir = replaceVariable(pseudoDir, 'JDFTx_pseudo') #Path to the pseudopotentials folder
                print(f"pseudo dir is {self.pseudoDir}")
                # self.pseudoDir = "/global/homes/b/beri9208/pseudopotentials"
                self.pseudoSet = pseudoSet
                # print(f"pseudo dir is {self.pseudoDir}")
                # self.pseudoDir = opj(pseudoDir, pseudoSet)

                if (self.executable is None):
                        raise Exception('Specify path to jdftx in argument \'executable\' or in environment variable \'JDFTx\'.')
                if (self.pseudoDir is None) and (not (pseudoSet in pseudoSetMap)):
                        raise Exception('Specify pseudopotential path in argument \'pseudoDir\' or in environment variable \'JDFTx_pseudo\', or specify a valid \'pseudoSet\'.')

                if pseudoSet in pseudoSetMap:
                        self.pseudoSetCmd = 'ion-species ' + pseudoSetMap[pseudoSet]
                else:
                        self.pseudoSetCmd = ''

                # Gets the input file template
                self.acceptableCommands = set(['electronic-SCF'])
                template = str(shell('%s -t' % (self.executable)))
                for match in re.findall(r"# (\S+) ", template):
                        self.acceptableCommands.add(match)

                self.input = [
                        ('dump-name', '$VAR'),
                        ('initial-state', '$VAR')
                ]
                # Nick edits
                self.InitialStateVars = []
                self.InitCommands = [('wavefunction','read wfns'),('elec-initial-fillings','read fillings'),
                                     ('elec-initial-eigenvals','eigenvals'),('fluid-initial-state','fluidState'),
                                     ('kpoint-reduce-inversion','yes')]


                # Parse commands, which can be a dict or a list of 2-tuples.
                if isinstance(commands, dict):
                        commands = commands.items()
                elif commands is None:
                        commands = []
                for cmd, v in commands:
                        self.addCommand(cmd, v)

                # Nick edits
                if ionic_steps != False and type(ionic_steps) == list:
                        self.addCommand('ionic-minimize', 'nIterations '+str(ionic_steps[0]) +
                                        ' energyDiffThreshold '+ str(ionic_steps[1]))

                # Nick edits
                if len(self.InitialStateVars) > 0:
                        self.input = [self.input[0]] + self.input[2:] # remove default initial-state when not wanted
                        for icmd in self.InitCommands:
                                continue
                                # add all default tags for standard operation that are not included directly
                                if icmd[0] not in self.InitialStateVars:
                                        self.addCommand(icmd[0], icmd[1])

                # Accepted pseudopotential formats
                self.pseudopotentials = ['fhi', 'uspp', 'upf', 'UPF', 'USPP']

                # Current results
                self.E = None
                self.Forces = None
                self.Charges = None

                # History
                self.atoms: Atoms | None = None
                self.lastInput = None

                # k-points
                self.kpoints = None

                # Dumps
                self.dumps = []
                #self.addDump("End", "State")
                self.addDump("End", "Forces")
                self.addDump("End", "Ecomponents")
                # self.addDump("End", "Dtot")
                # self.addDump("End", "ElecDensity")

                #Run directory:
                #self.runDir = tempfile.mkdtemp()
                self.runDir = calc_dir
                print('Set up JDFTx calculator with run files in \'' + self.runDir + '\'')

        ########### Interface Functions ###########

        def addCommand(self, cmd, v):
                # Nick edits
                if cmd in ['wavefunction','elec-initial-fillings','elec-initial-Haux','fluid-initial-state','kpoint-reduce-inversion']:
                        self.input.append((cmd, v))
                        self.InitialStateVars.append(cmd)
                        return

                if(not self.validCommand(cmd)):
                        if not debugging_override():
                                raise IOError('%s is not a valid JDFTx command!\nLook at the input file template (jdftx -t) for a list of commands.' % (cmd))
                self.input.append((cmd, v))

        def addDump(self, when, what):
                self.dumps.append((when, what))

        def addKPoint(self, pt, w=1):
                b1, b2, b3 = pt
                if self.kpoints is None:
                        self.kpoints = []
                self.kpoints.append((b1, b2, b3, w))

        def clean(self):
                shell('rm -rf ' + self.runDir)

        def calculation_required(self, atoms: Atoms, quantities):
                if((self.E is None) or (self.Forces is None)):
                        return True
                if((self.atoms != atoms) or (self.input != self.lastInput)):
                        return True
                return False

        def get_forces(self, atoms):
                if(self.calculation_required(atoms, None)):
                        self.update(atoms)
                return self.Forces

        def get_potential_energy(self, atoms, force_consistent=False):
                if(self.calculation_required(atoms, None)):
                        self.update(atoms)
                return self.E


        def get_charges(self, atoms):
                if (self.calculation_required(atoms, None)):
                        self.update(atoms)
                return self.Charges

        def get_stress(self, atoms):
                if self.ignoreStress:
                        return np.zeros((3, 3))
                else:
                        raise NotImplementedError(
                                'Stress calculation not implemented in JDFTx interface: set ignoreStress=True to ignore.')

        ################### I/O ###################


        def get_start_line(self, outfname):
                start = 0
                for i, line in enumerate(open(outfname)):
                        if "JDFTx 1." in line:
                                start = i
                return start


        ############## Running JDFTx ##############

        def update(self, atoms):
                self.runJDFTx(self.constructInput(atoms))
                

        def runJDFTx(self, inputfile):
                """ Runs a JDFTx calculation """
                #Write input file:
                fp = open(self.runDir+'/in', 'w')
                fp.write(inputfile)
                fp.close()
                #Run jdftx:
                print(f"Running in {self.runDir}")
                shell('cd %s && %s -i in -o out' % (self.runDir, self.executable))
                # print("DELETING FLUID-EX-CORR LINE FROM FUNCTIONAL - DELETE ME ONCE THIS BUG IS FIXED")
                # subprocess.run(f"sed -i '/fluid-ex-corr/d' {opj(self.runDir, 'out')}", shell=True, check=True)
                # subprocess.run(f"sed -i '/lda-PZ/d' {opj(self.runDir, 'out')}", shell=True, check=True)
                self._read_results()
                # print("reading energy")
                # self.E = self.__readEnergy('%s/Ecomponents' % (self.runDir))
                # print("reading forces")
                # self.Forces = self.__readForces('%s/force' % (self.runDir))
                # print("reading charges")
                # try:
                #         self.Charges = self.__readCharges('%s/out' % (self.runDir))
                # except Exception as e:
                #         print("The following error arose while trying to parse the charges;")
                #         print(e)
                #         pass

        def _read_results(self):
                outfile = JDFTXOutfile.from_file('%s/out' % (self.runDir))
                self.E = outfile.e
                self.Forces = outfile.forces
                self.Charges = outfile.structure.site_properties["charges"]
                self.Stress = outfile.stress
                self.Strain = outfile.strain
                self.results["energy"] = self.E
                self.results["forces"] = self.Forces
                self.results["charges"] = self.Charges
                self.results["stress"] = self.Stress
                self.results["strain"] = self.Strain

        def constructInput(self, atoms: Atoms):
                """ Constructs a JDFTx input string using the input atoms and the input file arguments (kwargs) in self.input """
                inputfile = ''

                # Add lattice info
                R = atoms.get_cell() / Bohr
                inputfile += 'lattice \\\n'
                for i in range(3):
                        for j in range(3):
                                inputfile += '%f  ' % (R[j, i])
                        if(i != 2):
                                inputfile += '\\'
                        inputfile += '\n'

                # Construct most of the input file
                inputfile += '\n'
                for cmd, v in self.input:
                        if '\\\\\\n' in v:
                                vc = '\\\n'.join(v.split('\\\\\\n'))
                        else:
                                vc = v + "\n"
                        inputfile += cmd + ' '
                        inputfile += vc + '\n'

                # Add ion info
                atomNames = atoms.get_chemical_symbols()   # Get element names in a list
                try:
                    fixed_atom_inds = atoms.constraints[0].get_indices()
                except:
                    fixed_atom_inds = []
                fixPos = []
                for i in range(len(atomNames)):
                    if i in fixed_atom_inds:
                        fixPos.append(0)
                    else:
                        fixPos.append(1)
                if self.direct:
                        atomPos = [x for x in list(atoms.get_scaled_positions())]
                        inputfile += '\ncoords-type lattice\n'
                else:
                        atomPos = [x / Bohr for x in list(atoms.get_positions())]  # Also convert to bohr
                        inputfile += '\ncoords-type cartesian\n'
                for i in range(len(atomNames)):
                        inputfile += 'ion %s %f %f %f \t %i\n' % (atomNames[i], atomPos[i][0], atomPos[i][1], atomPos[i][2], fixPos[i])
                del i

                # Add k-points
                if self.kpoints:
                        for pt in self.kpoints:
                                inputfile += 'kpoint %.8f %.8f %.8f %.14f\n' % pt

                #Add pseudopotentials
                inputfile += '\n'
                if not (self.pseudoDir is None):
                        pseudoSetDir = opj(self.pseudoDir, self.pseudoSet)
                        added = []  # List of pseudopotential that have already been added
                        for atom in atomNames:
                                if(sum([x == atom for x in added]) == 0.):  # Add ion-species command if not already added
                                        for filetype in self.pseudopotentials:
                                                try:
                                                        shell('ls %s | grep %s.%s' % (pseudoSetDir, atom, filetype))
                                                        inputfile += 'ion-species %s/%s.%s\n' % (pseudoSetDir, atom, filetype)
                                                        added.append(atom)
                                                        break
                                                except Exception as e:
                                                        print("issue calling pseudopotentials")
                                                        print(e)
                                                        pass
                inputfile += self.pseudoSetCmd + '\n' #Pseudopotential sets

                # Add truncation info (periodic vs isolated)
                inputfile += '\ncoulomb-interaction '
                pbc = list(atoms.get_pbc())
                if(sum(pbc) == 3):
                        inputfile += 'periodic\n'
                elif(sum(pbc) == 0):
                        inputfile += 'isolated\n'
                elif(sum(pbc) == 1):
                        inputfile += 'wire %i%i%i\n' % (pbc[0], pbc[1], pbc[2])
                elif(sum(pbc) == 2):
                        inputfile += 'slab %i%i%i\n' % (not pbc[0], not pbc[1], not pbc[2])
                #--- add truncation center:
                if(sum(pbc) < 3):
                        center = np.mean(np.array(atomPos), axis=0)
                        inputfile += 'coulomb-truncation-embed %g %g %g\n' % tuple(center.tolist())

                #Add dump commands
                inputfile += "".join(["dump %s %s\n" % (when, what) for when, what in self.dumps])

                # Cache this calculation to history
                self.atoms = atoms.copy()
                self.lastInput = list(self.input)
                return inputfile

        ############## JDFTx command structure ##############

        def validCommand(self, command):
                """ Checks whether the input string is a valid jdftx command \nby comparing to the input template (jdft -t)"""
                if(type(command) != str):
                        raise IOError('Please enter a string as the name of the command!\n')
                if command == 'ionic-minimize':
                    return True
                return command in self.acceptableCommands

        def help(self, command=None):
                """ Use this function to get help about a JDFTx command """
                if(command is None):
                        print('This is the help function for JDFTx-ASE interface. \
                                  \nPlease use the command variable to get information on a specific command. \
                                  \nVisit jdftx.sourceforge.net for more info.')
                elif(self.validCommand(command)):
                        raise NotImplementedError('Template help is not yet implemented')
                else:
                        raise IOError('%s is not a valid command' % (command))

class Wannier(Calculator):

        def __init__(self, executable=None, pseudoDir=None, pseudoSet='GBRV', commands=None, outfile=None, gpu=False):
                self.ran = False
                self.E = None
                print("initiating")
                #Valid pseudopotential sets (mapping to path and suffix):
                pseudoSetMap = {
                        'SG15' : 'SG15/$ID_ONCV_PBE.upf',
                        'GBRV' : 'GBRV/$ID_pbe.uspp',
        				'GBRV_v1.5' : 'GBRV_v1.5/$ID_pbe_v1.uspp',
                        'GBRV-pbe' : 'GBRV/$ID_pbe.uspp',
                        'GBRV-lda' : 'GBRV/$ID_lda.uspp',
                        'GBRV-pbesol' : 'GBRV/$ID_pbesol.uspp',
                        'kjpaw': 'kjpaw/$ID_pbe-n-kjpaw_psl.1.0.0.upf',
                        'dojo': 'dojo/$ID.upf',
                }

                #Get default values from environment:
                if gpu:
                        self.executable = replaceVariable(executable, 'Wannier_GPU')
                else:
                        self.executable = replaceVariable(executable, 'Wannier')
                self.pseudoDir = replaceVariable(pseudoDir, 'JDFTx_pseudo') #Path to the pseudopotentials folder
                print(f"pseudo dir is {self.pseudoDir}")
                self.pseudoSet = pseudoSet

                if (self.executable is None):
                        raise Exception('Specify path to jdftx in argument \'executable\' or in environment variable \'JDFTx\'.')
                if (self.pseudoDir is None) and (not (pseudoSet in pseudoSetMap)):
                        raise Exception('Specify pseudopotential path in argument \'pseudoDir\' or in environment variable \'JDFTx_pseudo\', or specify a valid \'pseudoSet\'.')

                if pseudoSet in pseudoSetMap:
                        self.pseudoSetCmd = 'ion-species ' + pseudoSetMap[pseudoSet]
                else:
                        self.pseudoSetCmd = ''

                # Gets the input file template
                self.acceptableCommands = set([])
                # self.acceptableCommands = set(['electronic-SCF'])
                template = str(shell('%s -t' % (self.executable)))
                for match in re.findall(r"# (\S+) ", template):
                        self.acceptableCommands.add(match)

                self.input = [
                        ('dump-name', '$VAR'),
                        ('initial-state', '$VAR')
                ]
                # Nick edits
                self.InitialStateVars = []
                self.InitCommands = [('kpoint-reduce-inversion','yes')]


                # Parse commands, which can be a dict or a list of 2-tuples.
                if isinstance(commands, dict):
                        commands = commands.items()
                elif commands is None:
                        commands = []
                for cmd, v in commands:
                        self.addCommand(cmd, v)

                # Accepted pseudopotential formats
                self.pseudopotentials = ['fhi', 'uspp', 'upf']

                # Current results
                self.E = None
                self.Charges = None

                # # History
                # self.lastAtoms = None
                # self.lastInput = None

                # k-points
                self.kpoints = None

                # Dumps
                self.dumps = []
                #self.addDump("End", "State")
                # self.addDump("End", "Forces")
                # self.addDump("End", "Ecomponents")
                # self.addDump("End", "Dtot")
                # self.addDump("End", "ElecDensity")

                #Run directory:
                #self.runDir = tempfile.mkdtemp()
                self.runDir = outfile
                print('Set up JDFTx calculator with run files in \'' + self.runDir + '\'')

        ########### Interface Functions ###########

        def addCommand(self, cmd, v):
                # Nick edits
                if cmd in ['wavefunction','elec-initial-fillings','elec-initial-Haux','fluid-initial-state','kpoint-reduce-inversion']:
                        self.input.append((cmd, v))
                        self.InitialStateVars.append(cmd)
                        return
                if(not self.validCommand(cmd)):
                        raise IOError('%s is not a valid JDFTx command!\nLook at the input file template (jdftx -t) for a list of commands.' % (cmd))
                self.input.append((cmd, v))

        def addDump(self, when, what):
                self.dumps.append((when, what))

        def addKPoint(self, pt, w=1):
                b1, b2, b3 = pt
                if self.kpoints is None:
                        self.kpoints = []
                self.kpoints.append((b1, b2, b3, w))

        def clean(self):
                shell('rm -rf ' + self.runDir)



        def calculation_required(self, atoms, quantities):
                return ((self.E is None) or (not self.ran))

        def get_potential_energy(self, atoms, force_consistent=False):
                try:
                        if(self.calculation_required(atoms, None)):
                                self.update(atoms)
                        return self.E
                except Exception as e:
                        print(e)

        #
        #
        # def get_charges(self, atoms):
        #         if (self.calculation_required(atoms, None)):
        #                 self.update(atoms)
        #         return self.Charges
        #
        # def get_stress(self, atoms):
        #         if self.ignoreStress:
        #                 return scipy.zeros((3, 3))
        #         else:
        #                 raise NotImplementedError(
        #                         'Stress calculation not implemented in JDFTx interface: set ignoreStress=True to ignore.')

        ################### I/O ###################

        def __readEnergy(self, filename):
                return 0
                # Efinal = None
                # for line in open(filename):
                #         tokens = line.split()
                #         if len(tokens)==3:
                #                 Efinal = float(tokens[2])
                # if Efinal is None:
                #         raise IOError('Error: Energy not found.')
                # return Efinal * Hartree #Return energy from final line (Etot, F or G)

        # def __readForces(self, filename):
        #         return
        #         idxMap = {}
        #         symbolList = self.lastAtoms.get_chemical_symbols()
        #         for i, symbol in enumerate(symbolList):
        #                 if symbol not in idxMap:
        #                         idxMap[symbol] = []
        #                 idxMap[symbol].append(i)
        #         forces = [0]*len(symbolList)
        #         for line in open(filename):
        #                 if line.startswith('force '):
        #                         tokens = line.split()
        #                         idx = idxMap[tokens[1]].pop(0) # tokens[1] is chemical symbol
        #                         forces[idx] = [float(word) for word in tokens[2:5]] # tokens[2:5]: force components
        #         if(len(forces) == 0):
        #                 raise IOError('Error: Forces not found.')
        #         return (Hartree / Bohr) * scipy.array(forces)


        # def __readCharges(self, filename):
        #         idxMap = {}
        #         symbolList = self.lastAtoms.get_chemical_symbols()
        #         for i, symbol in enumerate(symbolList):
        #                 if symbol not in idxMap:
        #                         idxMap[symbol] = []
        #                 idxMap[symbol].append(i)
        #         chargeDir={}
        #         key = "oxidation-state"
        #         start = self.get_start_line(filename)
        #         for i, line in enumerate(open(filename)):
        #                 if i > start:
        #                         if key in line:
        #                                 look = line.rstrip('\n')[line.index(key):].split(' ')
        #                                 symbol = str(look[1])
        #                                 charges = [float(val) for val in look[2:]]
        #                                 chargeDir[symbol] = charges
        #         charges = scipy.zeros(len(symbolList), dtype=float)
        #         for atom in list(chargeDir.keys()):
        #                 for i, idx in enumerate(idxMap[atom]):
        #                         charges[idx] += chargeDir[atom][i]
        #         return charges



        def get_start_line(self, outfname):
                start = 0
                for i, line in enumerate(open(outfname)):
                        if "JDFTx 1." in line:
                                start = i
                return start


        ############## Running JDFTx ##############

        def update(self, atoms):
                self.runWannier(self.constructInput(atoms))

        def runWannier(self, inputfile):
                """ Runs a JDFTx calculation """
                #Write input file:
                fp = open(self.runDir+'/in', 'w')
                fp.write(inputfile)
                fp.close()
                #Run jdftx:
                shell('cd %s && %s -i in -o out' % (self.runDir, self.executable))
                self.ran = True
                self.E = self.__readEnergy(None)
                # self.E = self.__readEnergy('%s/Ecomponents' % (self.runDir))
                # self.Forces = self.__readForces('%s/force' % (self.runDir))
                # self.Charges = self.__readCharges('%s/out' % (self.runDir))

        def constructInput(self, atoms: Atoms):
                """ Constructs a JDFTx input string using the input atoms and the input file arguments (kwargs) in self.input """
                inputfile = ''

                # Add lattice info
                R = atoms.get_cell() / Bohr
                inputfile += 'lattice \\\n'
                for i in range(3):
                        for j in range(3):
                                inputfile += '%f  ' % (R[j, i])
                        if(i != 2):
                                inputfile += '\\'
                        inputfile += '\n'

                # Construct most of the input file
                inputfile += '\n'
                for cmd, v in self.input:
                        if '\\\\\\n' in v:
                                vc = '\\\n'.join(v.split('\\\\\\n'))
                        else:
                                vc = v + "\n"
                        inputfile += cmd + ' '
                        inputfile += vc + '\n'

                # Add ion info
                atomPos = [x / Bohr for x in list(atoms.get_positions())]  # Also convert to bohr
                atomNames = atoms.get_chemical_symbols()   # Get element names in a list
                try:
                    fixed_atom_inds = atoms.constraints[0].get_indices()
                except:
                    fixed_atom_inds = []
                fixPos = []
                for i in range(len(atomPos)):
                    if i in fixed_atom_inds:
                        fixPos.append(0)
                    else:
                        fixPos.append(1)
                inputfile += '\ncoords-type cartesian\n'
                for i in range(len(atomPos)):
                        inputfile += 'ion %s %f %f %f \t %i\n' % (atomNames[i], atomPos[i][0], atomPos[i][1], atomPos[i][2], fixPos[i])
                del i

                # Add k-points
                if self.kpoints:
                        for pt in self.kpoints:
                                inputfile += 'kpoint %.8f %.8f %.8f %.14f\n' % pt

                #Add pseudopotentials
                inputfile += '\n'
                if not (self.pseudoDir is None):
                        pseudoSetDir = opj(self.pseudoDir, self.pseudoSet)
                        added = []  # List of pseudopotential that have already been added
                        for atom in atomNames:
                                if(sum([x == atom for x in added]) == 0.):  # Add ion-species command if not already added
                                        for filetype in self.pseudopotentials:
                                                try:
                                                        shell('ls %s | grep %s.%s' % (pseudoSetDir, atom, filetype))
                                                        inputfile += 'ion-species %s/%s.%s\n' % (pseudoSetDir, atom, filetype)
                                                        added.append(atom)
                                                        break
                                                except Exception as e:
                                                        print("issue calling pseudopotentials")
                                                        print(e)
                                                        pass
                inputfile += self.pseudoSetCmd + '\n' #Pseudopotential sets

                # Add truncation info (periodic vs isolated)
                inputfile += '\ncoulomb-interaction '
                pbc = list(atoms.get_pbc())
                if(sum(pbc) == 3):
                        inputfile += 'periodic\n'
                elif(sum(pbc) == 0):
                        inputfile += 'isolated\n'
                elif(sum(pbc) == 1):
                        inputfile += 'wire %i%i%i\n' % (pbc[0], pbc[1], pbc[2])
                elif(sum(pbc) == 2):
                        inputfile += 'slab %i%i%i\n' % (not pbc[0], not pbc[1], not pbc[2])
                #--- add truncation center:
                if(sum(pbc) < 3):
                        center = np.mean(np.array(atomPos), axis=0)
                        inputfile += 'coulomb-truncation-embed %g %g %g\n' % tuple(center.tolist())

                #Add dump commands
                inputfile += "".join(["dump %s %s\n" % (when, what) for when, what in self.dumps])

                # Cache this calculation to history
                self.lastAtoms = atoms.copy()
                self.lastInput = list(self.input)
                return inputfile

        ############## JDFTx command structure ##############

        def validCommand(self, command):
                """ Checks whether the input string is a valid jdftx command \nby comparing to the input template (jdft -t)"""
                if(type(command) != str):
                        raise IOError('Please enter a string as the name of the command!\n')
                if command == 'ionic-minimize':
                    return True
                return command in self.acceptableCommands

        def help(self, command=None):
                """ Use this function to get help about a JDFTx command """
                if(command is None):
                        print('This is the help function for JDFTx-ASE interface. \
                                  \nPlease use the command variable to get information on a specific command. \
                                  \nVisit jdftx.sourceforge.net for more info.')
                elif(self.validCommand(command)):
                        raise NotImplementedError('Template help is not yet implemented')
                else:
                        raise IOError('%s is not a valid command' % (command))