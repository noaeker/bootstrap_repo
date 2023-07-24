from sys import platform
import logging

if platform == "linux" or platform == "linux2":
    LOCAL_RUN = False
else:
    LOCAL_RUN = True

MODULE_LOAD_STR = "source /groups/itay_mayrose/lab_python/anaconda3/etc/profile.d/conda.sh; source activate noa_env;  module load gcc/gcc-7.2.0; module load R/3.6.1; module load mafft"
PBS_FILE_GENERATOR_CODE = "/groups/pupko/noaeker/q_submitter"

SEED = 5
BASELINE = "X"
CURR_RUN_PREFIX = "pandit_tuning"
CURR_JOBS_PREFIX = "pandit_tuning"
N_CPUS_PER_JOB = 1
N_CPUS_PER_TRAINING = 1
N_CPUS_RAXML = 1
N_MSAS =3
N_JOBS = 1
MIN_N_SEQ =30
MAX_N_SEQ = 200
N_PARSIMONY_RAXML_SEARCH = 3#15
N_RANDOM_RAXML_SEARCH = 0#15
MIN_N_LOCI = 1
LOGGING_LEVEL = "info" #"debug"
OUTPUT_CSV_NAME = "tune_raxml"
WAITING_TIME_UPDATE = 20
TEST_MSA_ITERATIONS = 30
EPSILON = 0.1
SPR_RADIUS_GRID =  ""#"1_30_10"
SPR_CUTOFF_GRID = ""#"0.1_10_10"
CSV_SEP = "\t"
CSV_SUFFIX = ".tsv"
N_MSAS_PER_BUNCH = -1
MSAs_POOL_SIZE = 1000





# PATH CONFIGURATION

if not LOCAL_RUN:
    IQTREE_EXE = "/groups/pupko/noaeker/programs/tree_search_programs/iqtree/bin/iqtree"
    IQTREE_SIM_PATH  = "/groups/pupko/noaeker/programs/other_programs/iqtree2"
    RAXML_NG_EXE = "/groups/pupko/noaeker/programs/tree_search_programs/raxml-ng/raxml-ng"
    RESULTS_FOLDER = "/groups/pupko/noaeker/bootstrap_results"
    RAxML_alisim_path="/groups/pupko/noaeker/RAxMLGroveScripts/org_script.py"
    MAIN_CODE_PATH_SIM = "/groups/pupko/noaeker/bootstrap_repo/simulations_generation/RAxML_grove_simulations.py"
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
    BOOSTER_EXE = "/groups/pupko/noaeker/programs/other_programs/booster_linux64"

elif LOCAL_RUN:
    IQTREE_EXE = "/Users/noa/Programs/iqtree-2.1.3-MacOSX/bin/iqtree2"
    IQTREE_SIM_PATH = "/Users/noa/Programs/iqtree-2.2.0-beta-MacOSX/bin/iqtree2"
    RAXML_NG_EXE = "/Users/noa/Programs/Raxml/raxml-ng"
    RESULTS_FOLDER = "/Users/noa/Workspace/bootstrap_results"
    RAxML_alisim_path = '/Users/noa/Workspace/RAxMLGroveScripts/org_script.py'
    MAIN_CODE_PATH_SIM = "/Users/noa/Workspace/bootstrap_repo/simulations_generation/RAxML_grove_simulations.py"
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
    BOOSTER_EXE = "/Users/noa/Programs/booster_macos64"
