from side_code.config import *
from side_code.code_submission import execute_command_and_write_to_log
import os
import subprocess
import re
import ast
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

def RAxML_grove_get_tree_statistics(out_dir, min_n_taxa, max_n_taxa, min_n_loci, max_n_loci, msa_type):
    msa_type = f"'{msa_type}'"
    cmd_raxml_grove = f'{RAxML_alisim_path} find --list -o {out_dir} -q " NUM_TAXA>={min_n_taxa} and NUM_TAXA<={max_n_taxa} and OVERALL_NUM_ALIGNMENT_SITES>={min_n_loci} and OVERALL_NUM_ALIGNMENT_SITES<={max_n_loci}  and DATA_TYPE=={msa_type}" '
    print(cmd_raxml_grove)
    logging.info(f'about to run {cmd_raxml_grove} to simulate tree')
    result = subprocess.run(cmd_raxml_grove,  stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, shell=True)
    all_tree_stats = result.stdout.decode("utf-8").split('\n')
    return pd.DataFrame([ast.literal_eval(tree_stats) for tree_stats in all_tree_stats[:-1]])




def extract_representative_sample(out_dir, min_n_taxa, max_n_taxa, min_n_loci, max_n_loci, msa_type, n_k_means_clusters):
    d = RAxML_grove_get_tree_statistics(out_dir, min_n_taxa, max_n_taxa, min_n_loci, max_n_loci, msa_type)
    k_means_data = d[["TREE_ID", "NUM_TAXA", "BRANCH_LENGTH_MEAN", "BRANCH_LENGTH_VARIANCE"]]
    scaler = preprocessing.StandardScaler()
    scaled_k_means_data = scaler.fit_transform(k_means_data.drop(columns=['TREE_ID']))
    kmeans = KMeans(n_clusters=n_k_means_clusters)
    kmeans_fit = kmeans.fit(scaled_k_means_data)
    k_means_data["label"] = kmeans_fit.labels_
    scaled_k_means_data_distances = (kmeans.transform(scaled_k_means_data)**2).sum(axis=1)
    k_means_data["dist_from_center"] = scaled_k_means_data_distances
    repersentative_sample = k_means_data.sort_values("dist_from_center").groupby("label").head(1).reset_index()
    return list(repersentative_sample["TREE_ID"])


def extract_model_specification_from_log(log_file, param):
    with open(log_file, 'r') as LOG_FILE:
        text = LOG_FILE.read()
        if param == 'model':
            match = re.search(r'- Model: (.*)', text)
            res = match.group(1)
        elif param == 'length':
            match = re.search(r'- Length of output sequences: (\d+)', text)
            res = int(match.group(1))
    return res


def RAxML_grove_tree_simulation( out_dir, min_n_taxa, max_n_taxa, min_n_loci, max_n_loci, msa_type, tree_id, model_modes):
    msa_type = f"'{msa_type}'"
    tree_id_cmd = "" if tree_id is None else f" and TREE_ID={tree_id}"
    cmd_raxml_grove = f'{RAxML_alisim_path} generate -g alisim -o {out_dir} -q " NUM_TAXA>={min_n_taxa} and NUM_TAXA<={max_n_taxa} and OVERALL_NUM_ALIGNMENT_SITES>={min_n_loci} and OVERALL_NUM_ALIGNMENT_SITES<={max_n_loci}  and DATA_TYPE=={msa_type} {tree_id_cmd}" '
    logging.info(f'about to run {cmd_raxml_grove} to simulate tree')
    execute_command_and_write_to_log(cmd_raxml_grove)
    logging.info("Done with simulations")
    folder = os.listdir(out_dir)[0]
    true_tree_path_orig = os.path.join(out_dir, folder, 'tree_best.newick')
    model_formulation_file = os.path.join(out_dir, folder, 'tree_best.newick.log')

    length = extract_model_specification_from_log(model_formulation_file, 'length')
    file_model = extract_model_specification_from_log(model_formulation_file, 'model')
    for file in os.listdir(os.path.join(out_dir, folder)):
        full_path = os.path.join(out_dir,folder,file)
        if full_path!=true_tree_path_orig:
            os.remove(full_path)

    base_sim_dict = {

                'length': length, 'tree_id': folder, 'tree_folder': os.path.join(out_dir, folder), 'tree_id': folder,
                'true_tree_path_orig': true_tree_path_orig}
    all_tree_sim_dicts = {}
    for model_mode in model_modes:
        sim_dict_copy = base_sim_dict.copy()
        if model_mode == 'standard':
            model = file_model
        elif model_mode == 'downgrade':
            model = 'GTR+F+G+I'
        else:
            model = 'JC'
        logging.info(f"Generating tree simulation dictionary for model mode is {model_mode} (using {model} model for simulation)")
        sim_dict_copy.update({'model_mode': model_mode,'simulation_model': model,'simulation_model_short': re.sub(r'\{[^{}]*\}','',model) if model_mode=='standard' else model})
        all_tree_sim_dicts[model_mode]=sim_dict_copy
    return all_tree_sim_dicts


