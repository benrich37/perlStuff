# perlStuff

This library holds functions meant for JDFTx calculations on perlMutter. These are meant to be a more hands on "step 2" from gc-manager calculations, so make sure that is running properly before running these. Additionally, make sure your python executable is at least v3.0, and has the ASE package installed.

NOTE: This repo is under heavy constructions, so pull updates often! (go to your repo clone directory and run `git pull origin`)

The main python scripts for this library (ie opt.py, scan_bond.py, se_neb.py, etc). In order to run these in the way intended, follow these instructions
1. Clone this repo onto your perlMutter home directory. (go to your home directory and run `git clone (ssh url for this repo)`)
2. Give the path to the perlStuff directory an environmental variable alias (ie for me, I have the line "export pstuff=$HOME/perlStuff" in my .bashrc, so that I can call
   opt.py for example as $pstuff/opt.py
3. Go to the directory created by gc-manager which contains the system you want to use one of these scripts on, and directly run the script through command line
   (ie `python $pstuff/opt.py`)
4. Running the python script directly is done to check for problems in the setup without wasting GPU/CPU time. If no specialized input file for the script ran is found, this first
   dry run will dump a template input file to the working directory for you to modify. These will have names like "opt_input" to differentiate them from the "inputs" file which
   these scripts use to gather calculation info for the JDFTx calculator. Modify this input file to your specifications
5. Run the python script directly again. This time, if all else is good, the script will check for a "psubmit.sh" slurm file in the working directory. If none is found, a template
   one will be dumped to the working directory and the script will be aborted before trying to run the DFT calculation. Again, change this to your specifications (how much time
   to give the calculation, how much memory, etc).
6. Run the python script directly another time. If there are any other issues with your setup, they will be exposed in this process. Otherwise, you will see a message printed
   indicating that your caclulation is ready for submitting.
7. Check the .iolog file (ie opt.iolog) for info on how the dry run performed the calculation setup to make sure everything went according to your liking.
8. Submit the slurm script (`sbatch psubmit.sh`)
9. Check the .iolog file for progress updates. This file contains a verbose log of what the script is doing along with timestamps. Unfortunately, failed calculations
   within these scripts will appear as "COMPLETED" instead of "FAILED", so manual results inspection is necessary.

This library makes heavy use of ".logx" files for showing optimizations to the user. These can be opened in gaussview. These will occasionally also store the Lowdin charges
(called mulliken charges so that gaussview can parse them), but otherwise these contain the same information as the .traj files we usually look at. 
