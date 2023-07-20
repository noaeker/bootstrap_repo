from side_code.config import *
from side_code.file_handling import create_or_clean_dir, delete_dir_content, unify_text_files
from side_code.code_submission import execute_command_and_write_to_log
from side_code.basic_trees_manipulation import get_tree_string, generate_multiple_tree_object_from_newick_file
import os
import time
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




def iqtree_search(curr_run_directory, msa_path, msa_type,prefix,  starting_tree_path):

    starting_trees_command = f"-t {starting_tree_path}"
    if LOCAL_RUN:
        starting_trees_command+=" -n 0 "
    search_prefix = os.path.join(curr_run_directory, prefix)
    model = "GTR+G" if msa_type == "DNA" else "WAG+G"
    search_command = f"{IQTREE_EXE} -s {msa_path} -nt 1 -m {model} {starting_trees_command} -seed {SEED} -pre {search_prefix} -redo "
    iqtree_general_log_file = search_prefix + ".log"
    iqtree_log_file = search_prefix + ".iqtree"
    execute_command_and_write_to_log(search_command, print_to_log=True)
    best_ll = extract_param_from_IQTREE_log(iqtree_log_file, 'll')
    starting_tree_ll = extract_param_from_IQTREE_log(iqtree_general_log_file, 'starting_tree_ll')
    best_tree_topology_path = search_prefix + ".treefile"
    res = {'spr_radius': -1, 'spr_cutoff': -1, 'final_ll': best_ll,
           'starting_tree_ll': starting_tree_ll,
           'elapsed_running_time': -1,
           'final_tree_topology': get_tree_string(best_tree_topology_path)}
    print(res)
    return res





def iqtree_ll_eval(curr_run_directory, msa_path, msa_type,prefix,  starting_tree_path):

    starting_trees_command = f"-t {starting_tree_path}"
    search_prefix = os.path.join(curr_run_directory, prefix)
    model = "GTR+G" if msa_type == "DNA" else "WAG+G"
    search_command = f"{IQTREE_EXE} -s {msa_path} -nt 1 -n 0 -blfix -m {model} {starting_trees_command} -seed {SEED} -pre {search_prefix} -redo "
    iqtree_log_file = search_prefix + ".iqtree"
    execute_command_and_write_to_log(search_command, print_to_log=True)
    best_ll = extract_param_from_IQTREE_log(iqtree_log_file, 'll')
    return best_ll




