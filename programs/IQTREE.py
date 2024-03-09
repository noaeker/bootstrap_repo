from side_code.config import *
from side_code.file_handling import create_or_clean_dir, delete_dir_content, unify_text_files
from side_code.code_submission import execute_command_and_write_to_log
from side_code.basic_trees_manipulation import get_tree_string, generate_multiple_tree_object_from_newick_file
import os
import time
from side_code.basic_trees_manipulation import *
import re
# from side_code.basic_trees_manipulation import *
import datetime


class GENERAL_IQTREE_ERROR(Exception):
    pass

def extract_param_from_IQTREE_log(iqtree_log_path, param_name, raise_error=True):
    with open(iqtree_log_path) as iqtree_log_file:
        logging.debug(f"Opened succesfully file {iqtree_log_path}")
        data = iqtree_log_file.read()
        if (param_name == "ll"):
                pattern = r'Log-likelihood of the tree: (-[\d.]+)'
                match = re.search(pattern, data, re.IGNORECASE)
        elif (param_name == "starting_tree_ll"):
                pattern = r'Optimal log-likelihood: (-[\d.]+)'
                match = re.search(pattern, data, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value
        else:
            error_msg = "Param {param_name} not found in file".format(param_name=param_name)
            if raise_error:
                raise GENERAL_IQTREE_ERROR(error_msg)
            else:
                return None




def iqtree_bootstrap_search(curr_run_directory, msa_path, model,nb, prefix, n_cpus = 1):

    search_prefix = os.path.join(curr_run_directory, prefix)
    st = time.time()
    search_command = f"{IQTREE_EXE} -s {msa_path} -nt {n_cpus}  -m {model} -seed {SEED} -pre {search_prefix} -alrt 0 -abayes -bb {nb} -redo "
    best_tree_topology_path = search_prefix + ".treefile"
    execute_command_and_write_to_log(search_command, print_to_log=True)
    end = time.time()
    return best_tree_topology_path, end-st


def iqtree_standard_search_running_time(curr_run_directory, msa_path, model,prefix):

    search_prefix = os.path.join(curr_run_directory, prefix)
    st = time.time()
    search_command = f"{IQTREE_EXE} -s {msa_path}  -m {model} -seed {SEED} -pre {search_prefix}  -redo "
    execute_command_and_write_to_log(search_command, print_to_log=True)
    end = time.time()
    return end-st



def diffrentiate_bootstrap_trees(t):
    with open(t) as T:
        text = T.read()
    pattern = re.compile(r'/[\d.]+/[\d.]+/[\d.]+')
    res = [text, text, text]
    for i in range(3):
        for match in re.findall(pattern, text):
            m_tuple = match.split("/")[1:]
            res[i] = res[i].replace(match, m_tuple[i])
    return {'final_tree_aLRT':res[0],'final_tree_aBayes': res[1], 'final_tree_ultrafast': res[2]}


def iqtree_pipeline(curr_run_directory,results_folder, msa_path, model,nb,  prefix, n_cpus):
    ML_tree, full_running_time = iqtree_bootstrap_search(curr_run_directory, msa_path, model,nb,n_cpus, prefix)
    standard_running_time = iqtree_standard_search_running_time(curr_run_directory, msa_path, model,prefix = prefix+"_standard")
    bootstrap_running_time = full_running_time-standard_running_time
    different_tree_topologies =  diffrentiate_bootstrap_trees(ML_tree)
    res = {'boot_run_time': bootstrap_running_time}
    for tree_type in different_tree_topologies:
        path = os.path.join(results_folder,f'{tree_type}.tree')
        with open(path,'w') as T:
            T.write(different_tree_topologies[tree_type])
        res[tree_type] = path
    return res




#t = iqtree_search(curr_run_directory = os.path.join(os.getcwd(),'trash'), msa_path = "/Users/noa/Workspace/bootstrap_results/test/job_0/raxml_tree_0/25117/iqtree_msa_0/sim_msa.fa",model = 'GTR+G',nb = 1000,prefix = 'prefix')
#t = "/Users/noa/Workspace/bootstrap_repo/side_code/trash/prefix.treefile"
#text = ""
#print(diffrentiate_bootstrap_trees(t))
