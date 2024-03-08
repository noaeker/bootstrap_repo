
from side_code.config import *
from side_code.file_handling import create_or_clean_dir, delete_dir_content, unify_text_files
from side_code.code_submission import execute_command_and_write_to_log
from side_code.basic_trees_manipulation import get_tree_string, generate_multiple_tree_object_from_newick_file,generate_multiple_newicks_from_newick_file
from side_code.MSA_manipulation import get_alignment_data
import os
import shutil
import time
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import re
# from side_code.basic_trees_manipulation import *
import datetime




class GENERAL_RAXML_ERROR(Exception):
    pass






def extract_param_from_raxmlNG_log(raxml_log_path, param_name, raise_error=True):
    with open(raxml_log_path) as raxml_log_file:
        logging.debug(f"Opened succesfully file {raxml_log_path}")
        data = raxml_log_file.read()
        if (param_name == "alpha"):
            pattern = r'alpha: ([\d.]+)'
        elif (param_name == "search_ll"):
            pattern = r'Final LogLikelihood: (-[\d.]+)'
        elif (param_name == "rf_dist"):
            pattern = 'Average relative RF distance in this tree set: ([\d.]+)'
        elif (param_name == "time"):
            pattern = 'Elapsed time: ([\d.]+)'
        elif (param_name == "spr_cutoff"):
            pattern = 'spr subtree cutoff: ([\d.]+)'
        elif (param_name == "n_parsimony"):
            pattern = 'parsimony \((\d+)\)'
        elif (param_name == "n_random"):
            pattern = 'start tree\(s\): random \((\d+)\)'
        elif (param_name == "starting_tree_ll"):
            pattern = '\[\d{2}:\d{2}:\d{2} (-[\d.]+)\] Model parameter optimization'
        elif (param_name == "spr_radius"):
            pattern = 'SPR radius for FAST iterations: ([\d]+) '
            value_strings = re.findall(pattern, data)
            value_floats = [float(ll) for ll in value_strings]
            if len(value_floats) == 1:
                return value_floats[0]
            else:
                # logging.info("{} ll values were extracted from log file".format(len(value_floats)))
                return value_floats
        elif (param_name == "ll"):
            pattern = r'Tree #\d+, final logLikelihood: (-[\d.]+)'
            value_strings = re.findall(pattern, data)
            value_floats = [float(ll) for ll in value_strings]
            if len(value_floats) == 1:
                return value_floats[0]
            else:
                # logging.info("{} ll values were extracted from log file".format(len(value_floats)))
                return value_floats
        match = re.search(pattern, data, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value
        else:
            error_msg = "Param {param_name} not found in file".format(param_name=param_name)
            if raise_error:
                raise GENERAL_RAXML_ERROR(error_msg)
            else:
                return None


def wait_for_file_existence(path, name):
    if not os.path.exists(path):
        # logging.info("{name} was succesfully created in: {path}".format(name=name, path=path))
        error_msg = "{name} was not generated in: {path}".format(name=name, path=path)
        logging.error(error_msg)
        start_time = time.time()
        while not os.path.exists(path):
            time.sleep(WAITING_TIME_UPDATE)
            logging.info("current time {}: file {} does not exist yet in path {}".format(datetime.now(), name, path))
            time.sleep(WAITING_TIME_UPDATE)
            if time.time() - start_time > 3600 * 24:
                logging.info("Waiting to much for param {}, breaking".format(name))
                break
        raise GENERAL_RAXML_ERROR(error_msg)


def get_unique_trees_mapping(curr_run_directory, trees):
    get_unique_trees_dir = os.path.join(curr_run_directory, "get_unique_trees_mapping")
    create_or_clean_dir(get_unique_trees_dir)
    all_trees_path = os.path.join(get_unique_trees_dir,"current_msa_trees")
    with open(all_trees_path,'w') as TREES_PATH:
        TREES_PATH.writelines(trees)
    rf_prefix = os.path.join(get_unique_trees_dir, "unique_topologies_filter")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=all_trees_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    rf_distances_file_path = rf_prefix + ".raxml.rfDistances"
    with open(rf_distances_file_path, 'r') as DIST:
        distances = DIST.readlines()
        clusters_keys= []
        trees_mapping = {}
        for line in distances:
            lst = line.split("\t")
            curr_tree, comp_tree, dist = int(lst[0]), int(lst[1]), int(lst[2])
            if curr_tree not in clusters_keys and curr_tree not in trees_mapping: #Add new cluster if needed
                clusters_keys.append(curr_tree)
            if dist==0 and comp_tree not in trees_mapping:
                trees_mapping[comp_tree] = curr_tree
        trees_mapping.update({tree_ind: tree_ind for tree_ind in clusters_keys})
        last_tree = len(trees)-1
        if last_tree not in trees_mapping:
            trees_mapping[last_tree] = last_tree
    return trees_mapping


def filter_unique_topologies(curr_run_directory, trees_path, n):
    logging.debug("Removing duplicate SPR neighbours")
    rf_prefix = os.path.join(curr_run_directory, "SPR_neighbours")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    rf_distances_file_path = rf_prefix + ".raxml.rfDistances"
    unique_file_path = trees_path + "_unique"
    unique_topology_inds = set(list(range(n)))
    with open(rf_distances_file_path, 'r') as DIST, open(trees_path, 'r') as TREES, open(unique_file_path,
                                                                                         'w') as UNIQUE_TREES:
        distances = DIST.readlines()
        original_trees = TREES.readlines()
        for line in distances:
            lst = line.split("\t")
            curr_tree, comp_tree, dist = int(lst[0]), int(lst[1]), int(lst[2])
            if curr_tree in unique_topology_inds and comp_tree in unique_topology_inds and dist == 0:
                unique_topology_inds.remove(comp_tree)
        unique_trees = [original_trees[ind] for ind in unique_topology_inds]
        UNIQUE_TREES.writelines(unique_trees)
    rf_prefix = os.path.join(curr_run_directory, "SPR_neighbours_check")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=unique_file_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    return unique_file_path


def generate_n_unique_tree_topologies_as_starting_trees(n, original_file_path, curr_run_directory,
                                                        seed, tree_type, model):
    trees_path = generate_n_tree_topologies(n, original_file_path, curr_run_directory,
                                            seed, tree_type, model)
    if tree_type == "pars" and n > 1:
        rf_prefix = os.path.join(curr_run_directory, "parsimony_rf_eval")
        rf_command = (
            "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
            raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
        execute_command_and_write_to_log(rf_command)
        rf_distances_file_path = rf_prefix + ".raxml.rfDistances"
        trees_path = extract_parsimony_unique_topologies(curr_run_directory, trees_path,
                                                         rf_distances_file_path, n)
    with open(trees_path) as trees_path:
        newicks = trees_path.read().split("\n")
    return newicks


def generate_n_tree_topologies(n, msa_path, curr_run_directory,
                               seed, tree_type, model):
    prefix = os.path.join(curr_run_directory, f"{tree_type}_tree_generation")
    random_tree_generation_command = (
        "{raxml_exe_path} --force msa --force perf_threads  --msa {msa_path} --model {model} --start --tree {tree_type}{{{n}}} --prefix {prefix} --seed {seed} --redo ").format(
        n=n, raxml_exe_path=RAXML_NG_EXE, tree_type=tree_type,
        msa_path=msa_path, prefix=prefix, seed=seed, model=model)
    trees_path = prefix + ".raxml.startTree"
    execute_command_and_write_to_log(random_tree_generation_command)
    return trees_path


def extract_parsimony_unique_topologies(curr_run_directory, trees_path, dist_path, n):
    rf_prefix = os.path.join(curr_run_directory, "parsimony_rf")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    unique_file_path = trees_path + "_unique"
    unique_topology_inds = set(list(range(n)))
    with open(dist_path, 'r') as DIST, open(trees_path, 'r') as TREES, open(unique_file_path, 'w') as UNIQUE_TREES:
        distances = DIST.readlines()
        original_trees = TREES.readlines()
        for line in distances:
            lst = line.split("\t")
            curr_tree, comp_tree, dist = int(lst[0]), int(lst[1]), int(lst[2])
            if curr_tree in unique_topology_inds and comp_tree in unique_topology_inds and dist == 0:
                unique_topology_inds.remove(comp_tree)
        unique_trees = [original_trees[ind] for ind in unique_topology_inds]
        n_unique_top = len(unique_trees)
        UNIQUE_TREES.writelines(unique_trees)
    rf_prefix = os.path.join(curr_run_directory, "parsimony_check_rf")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=unique_file_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    return unique_file_path



def extract_bootstrap_running_time(raxml_log_path):
    with open(raxml_log_path) as raxml_log_file:
        data = raxml_log_file.read()
    start_pattern = r'\[(\d{2,}:\d{2,}:\d{2,})\] Starting bootstrapping analysis with'
    end_pattern = r'\[(\d{2,}:\d{2,}:\d{2,})\] Bootstrapping converged after'
    start_time = re.search(start_pattern,data).group(1)
    try:
        end_time = re.search(end_pattern, data).group(1)
    except Exception as E:
        end_pattern_1000 = r'\[(\d{2,}:\d{2,}:\d{2,})\] Bootstrap tree #1000'
        end_time_match = re.search(end_pattern_1000, data)
        end_time = -1 if end_time_match is None else end_time_match.group(1)
    try:
        start_time_hours, start_time_min, start_time_sec = [int(t) for t in start_time.split(':')]
        end_time_hours,end_time_min, end_time_sec = [int(t) for t in end_time.split(':')]
        total_seconds = (end_time_sec-start_time_sec)+(end_time_min-start_time_min)*60+(end_time_hours-start_time_hours)*3600
    except:
        total_seconds = -1
    return total_seconds




def remove_redundant_sequences(curr_run_directory, prefix,msa_path, model, out_msa_path = None):
    if out_msa_path==None:
        out_msa_path = msa_path
    search_prefix = os.path.join(curr_run_directory, prefix)
    search_command = (
        "{raxml_exe} --check --msa {msa_path} --model {model}  --seed {seed} --prefix {prefix} --redo").format(
        raxml_exe=RAXML_NG_EXE,
        msa_path=msa_path, seed=SEED,
        prefix=search_prefix, model=model)
    execute_command_and_write_to_log(search_command, print_to_log=True)
    reduced_path = search_prefix + ".raxml.reduced.phy"
    if os.path.exists(reduced_path):
        logging.info("Generating a reduced alignment")
        all_msa_records = get_alignment_data(reduced_path)
        SeqIO.write(all_msa_records, out_msa_path, 'fasta')
    else:
        if msa_path!=out_msa_path:
            shutil.copyfile(msa_path,out_msa_path)
    for file in os.listdir(curr_run_directory):
        if prefix in file:
            os.remove(os.path.join(curr_run_directory,file))

def raxml_bootstrap_pipeline(curr_run_directory, results_folder, msa_path, prefix, model, n_cpus = 1, n_workers ='auto'):

    search_prefix = os.path.join(curr_run_directory, prefix)
    search_command = (
        "{raxml_exe} --blopt nr_safe --all --bs-trees autoMRE --threads {n_cpus} --workers {n_workers}  --force msa --force perf_threads --msa {msa_path} --model {model}  --seed {seed} --prefix {prefix} --redo").format(raxml_exe=RAXML_NG_EXE,
        msa_path=msa_path,  seed=SEED,
        prefix=search_prefix, model=model, n_cpus = n_cpus, n_workers = n_workers)
    raxml_log_file = search_prefix + ".raxml.log"
    model_file = search_prefix+".raxml.bestModel"
    execute_command_and_write_to_log(search_command, print_to_log=True)
    with open(model_file,'r') as MODEL:
        model_str = MODEL.read().split(',')[0]
    bootstrap_run_time = extract_bootstrap_running_time(raxml_log_file)
    best_tree_topology_path_orig = search_prefix + ".raxml.support"
    all_final_trees_path_orig = search_prefix + ".raxml.mlTrees"

    #model_str = re.sub('\+FU\{[^{}]*\}', '', model_str)
    #model_str = model_str.replace('4m', '')
    final_tree_topology_path = os.path.join(results_folder,'raxml_final_tree_topology.tree')
    shutil.move(best_tree_topology_path_orig,final_tree_topology_path)
    all_final_tree_topologies_path = os.path.join(results_folder, 'raxml_all_final_tree_topologies.tree')
    shutil.move(all_final_trees_path_orig,all_final_tree_topologies_path)
    res = {
           'final_tree_topology_path': final_tree_topology_path, 'all_final_tree_topologies_path':all_final_tree_topologies_path,'program_boot_run_time':bootstrap_run_time}
    return res


def standard_raxml_search(curr_run_directory, msa_path, prefix,model,n_pars, n_rand, n_cpus = 1, n_workers = 'auto'):

    search_prefix = os.path.join(curr_run_directory, prefix)
    search_command = (
        "{raxml_exe_path} --threads {n_cpus} --workers {n_workers} --tree pars{{{n_pars}}},rand{{{n_rand}}}  --force msa --force perf_threads --msa {msa_path} --model {model} --seed {seed} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE,
        msa_path=msa_path,  seed=SEED, n_pars = n_pars, n_rand = n_rand,
        prefix=search_prefix, model=model, n_cpus = n_cpus, n_workers = n_workers)
    print(search_command)
    raxml_log_file = search_prefix + ".raxml.log"
    model_file = search_prefix+".raxml.bestModel"
    best_tree_topology_path = search_prefix + ".raxml.bestTree"
    all_final_trees_path = search_prefix + ".raxml.mlTrees"
    if not  os.path.exists(best_tree_topology_path):
        logging.info("Running RAxML from beggining")
        execute_command_and_write_to_log(search_command, print_to_log=True)
    else:
        logging.info("Using existing RAxML files")
    with open(model_file,'r') as MODEL:
        model_str = MODEL.read().split(',')[0]
    elapsed_running_time = extract_param_from_raxmlNG_log(raxml_log_file, 'time')
    best_ll = extract_param_from_raxmlNG_log(raxml_log_file, 'search_ll')
    starting_tree_ll = extract_param_from_raxmlNG_log(raxml_log_file, 'starting_tree_ll')


    model_str = re.sub('\+FU\{[^{}]*\}', '', model_str)
    model_str = model_str.replace('4m', '')

    res = {'final_ll': best_ll,
           'starting_tree_ll': starting_tree_ll,
           'elapsed_running_time': elapsed_running_time,
           'inferred_model' : model_str,
           'final_tree_topology': get_tree_string(best_tree_topology_path),
           'final_tree_topology_path': best_tree_topology_path,
           'all_final_trees_path': all_final_trees_path,
           'all_final_tree_topologies': generate_multiple_newicks_from_newick_file(all_final_trees_path)
           }
    return res


def raxml_optimize_trees_for_given_msa(full_data_path, ll_on_data_prefix, tree_file,
                                       curr_run_directory, model, opt_model_and_brlen=True,n_cpus = 1, n_workers = 'auto', return_opt_tree = False,
                                       program_path = None):
    curr_run_directory = os.path.join(curr_run_directory, ll_on_data_prefix)
    if os.path.exists(curr_run_directory):
        delete_dir_content(curr_run_directory)
    else:
        os.mkdir(curr_run_directory)
    prefix = os.path.join(curr_run_directory, ll_on_data_prefix)
    brlen_command = "--opt-branches off --opt-model off " if not opt_model_and_brlen else ""
    if program_path is None:
        program_path = RAXML_NG_EXE
    compute_ll_run_command = (
        "{raxml_exe_path} --force msa --threads {n_cpus} --workers {n_workers} --workers auto --evaluate --msa {msa_path} --model {model} {brlen_command} --tree {tree_file} --seed {seed} --prefix {prefix} --redo").format(
        raxml_exe_path= program_path, msa_path=full_data_path, tree_file=tree_file, seed=SEED,
        prefix=prefix, brlen_command=brlen_command, model=model, n_cpus = n_cpus, n_workers = n_workers)
    #optimized_trees_path = prefix + ".raxml.mlTrees"
    best_tree_path = prefix + ".raxml.bestTree"
    raxml_log_file = prefix + ".raxml.log"
    execute_command_and_write_to_log(compute_ll_run_command)
    trees_ll_on_data = extract_param_from_raxmlNG_log(raxml_log_file, "ll")
    #tree_alpha = extract_param_from_raxmlNG_log(raxml_log_file, "alpha")
    #optimized_trees_final_path = optimized_trees_path if os.path.exists(optimized_trees_path) else best_tree_path
    if return_opt_tree:
        return trees_ll_on_data,best_tree_path
    else:
        return trees_ll_on_data#, tree_alpha, optimized_trees_final_path

def raxml_extract_sitelh(sitelh_file):
    logging.debug("Extracting sitelh from sitelh_file in {}".format(sitelh_file))
    with open(sitelh_file) as SITELH:
        all_lines = SITELH.readlines()
        sitelh = (re.sub(r'tree1', '', all_lines[1])).strip()
        sitelh_list = sitelh.split(" ")
        # print(sitelh_list)
        sitelh_list_float = [float(ll) for ll in sitelh_list]
        return (sitelh_list_float)


def raxml_compute_tree_per_site_ll(curr_run_directory, full_data_path, tree_file, ll_on_data_prefix, model, opt = True):
    compute_site_ll_prefix = os.path.join(curr_run_directory, ll_on_data_prefix)
    opt_model_cmd = ""
    if not opt:
        opt_model_cmd+=" --opt-branches off --opt-model off "
    compute_site_ll_run_command = (
            "{raxml_exe_path} --sitelh --msa {msa_path} --threads 1 --force perf_threads --model {model} {opt_model_cmd} --tree {tree_file} --prefix {prefix}  --redo").format(raxml_exe_path =RAXML_NG_EXE,
        model=model, msa_path=full_data_path, tree_file=tree_file,
        prefix=compute_site_ll_prefix,opt_model_cmd=opt_model_cmd)
    execute_command_and_write_to_log( compute_site_ll_run_command)
    sitelh_file = compute_site_ll_prefix + ".raxml.siteLH"
    sitelh_list = raxml_extract_sitelh(sitelh_file)
    return sitelh_list


def RF_distances(curr_run_directory, trees_path_a, trees_path_b=None, name="RF"):
    rf_prefix = os.path.join(curr_run_directory, name)
    trees_path = trees_path_a + (f",{trees_path_b}" if trees_path_b else "")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    rf_log_file_path = rf_prefix + ".raxml.rfDistances"
    rf_distances = []
    with open(rf_log_file_path) as RF:
        distances = RF.readlines()
        for line in distances:
            lst = line.split("\t")
            curr_tree, comp_tree, dist = int(lst[0]), int(lst[1]), int(lst[2])
            rf_distances.append(dist)
    return rf_distances


def calculate_rf_dist(rf_file_path, curr_run_directory, prefix="rf"):
    rf_prefix = os.path.join(curr_run_directory, prefix)
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix} --redo").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=rf_file_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    rf_log_file_path = rf_prefix + ".raxml.log"
    relative_rf_dist = extract_param_from_raxmlNG_log(rf_log_file_path, "rf_dist")
    return relative_rf_dist


def is_plausible_set_by_iqtree(tree_test_log_file):
    with open(tree_test_log_file) as iqtree_log_file:
        data = iqtree_log_file.readlines()
        for i,line in enumerate(data):
            if line.split()==['Tree','logL','deltaL','bp-RELL','p-KH','p-SH','p-WKH','p-WSH','c-ELW','p-AU']:
                break
        relevant_line = data[i+3].split()
        return {'au_test_sign':relevant_line[-1], 'au_test':relevant_line[-2],
                'wsh_test_sign':relevant_line[-5],'wsh_test':relevant_line[-6],
                'wkh_test_sign':relevant_line[-7],'wkh_test':relevant_line[-8],
                'sh_test_sign': relevant_line[-9], 'sh_test': relevant_line[-10],
                'kh_test_sign': relevant_line[-11], 'kh_test': relevant_line[-12]

                }





def rf_distance(curr_run_directory, tree_str_a, tree_str_b, name=f"rf_calculations"):
    rf_folder = os.path.join(curr_run_directory, name)
    create_or_clean_dir(rf_folder)
    rf_output_path = os.path.join(rf_folder, "rf_calculations")
    rf_first_phase_trees = unify_text_files([tree_str_a, tree_str_b], rf_output_path, str_given=True)
    rf = calculate_rf_dist(rf_first_phase_trees, rf_folder,
                           prefix="rf_calculations")
    return rf



def min_rf_distance(curr_run_directory, tree_str_a, tree_list_b, name=f"rf_calculations"):
    rf_folder = os.path.join(curr_run_directory, name)
    create_or_clean_dir(rf_folder)
    rf_output_path = os.path.join(rf_folder, "rf_calculations")
    rf_values = []
    for tree_str_b in tree_list_b:
        trees_file = unify_text_files([tree_str_a, tree_str_b], rf_output_path, str_given=True)
        rf = calculate_rf_dist(trees_file, rf_folder,
                               prefix="rf_calculations")
        rf_values.append(rf)
    return np.min(rf)


def au_test(curr_run_directory, per_tree_clusters_data, ML_tree, msa_path, cpus_per_job,msa_type, name=f"sh_calculations"):
    sh_folder = os.path.join(curr_run_directory, name)
    create_or_clean_dir(sh_folder)
    sh_test_output_path = os.path.join(sh_folder, "sh_test_tree")
    sh_run_folder = os.path.join(curr_run_directory, "sh_run")
    res_vec = []
    model = "GTR+G" if msa_type == "DNA" else "WAG+G"
    for ind,row in per_tree_clusters_data.iterrows():
        tree = row["final_tree_topology"]
        cluster_ind = row["tree_clusters_ind"]
        create_or_clean_dir(sh_run_folder)
        unify_text_files([ML_tree] + [tree], sh_test_output_path, str_given=True)
        sh_prefix = os.path.join(sh_run_folder, "curr_sh_run")
        sh_command = f'{IQTREE_EXE} -s {msa_path} -z {sh_test_output_path} -n 0 -zb 10000 -zw -au -pre {sh_prefix} -m {model}'
        if not LOCAL_RUN:
            sh_command = sh_command + f' -nt {cpus_per_job}'
        execute_command_and_write_to_log(sh_command)
        log_file = sh_prefix + ".iqtree"
        res = is_plausible_set_by_iqtree(log_file)
        res.update({"tree_clusters_ind": cluster_ind})
        res_vec.append(res)
    return res_vec



