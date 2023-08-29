import sys
import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from side_code.config import *
from side_code.basic_trees_manipulation import get_tree_string
from side_code.code_submission import execute_command_and_write_to_log
from side_code.raxml import raxml_bootstrap_pipeline
from side_code.FastTree import fasttree_pipeline
from side_code.IQTREE import iqtree_pipeline
from simulation_edit.bootstrap_edit import msa_path_bootstrap_analysis
from simulations_generation.simulations_argparser import  job_parser
from simulations_generation.msa_features import get_msa_stats
from simulation_edit.bootstrap_edit import msa_path_bootstrap_analysis
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir
import sys
import pandas as pd
import os
import re
import time



def simulate_msa(output_prefix, tree_file, model, length, ret_cmd = False):
    '''

    :param output_prefix:
    :param model: WAG / GTR
    :param treefile:
    :return:
    '''
    command = f"{IQTREE_SIM_PATH} --alisim {output_prefix} -m {model} -t {tree_file}  --length {int(length)} --out-format fasta" \
        f"  "
    # logging.info(f"About to run {command}")
    execute_command_and_write_to_log(command)
    msa_path = f"{output_prefix}.fa"
    if ret_cmd:
        return msa_path, command
    else:
        return msa_path


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


#and OVERALL_NUM_PARTITIONS==1


def RAxML_grove_tree_simulation( out_dir, min_n_taxa, max_n_taxa, min_n_loci, max_n_loci, msa_type):
    msa_type = f"'{msa_type}'"
    cmd_raxml_grove = f'{RAxML_alisim_path} generate -g alisim -o {out_dir} -q " NUM_TAXA>={min_n_taxa} and NUM_TAXA<={max_n_taxa} and OVERALL_NUM_ALIGNMENT_SITES>={min_n_loci} and OVERALL_NUM_ALIGNMENT_SITES<={max_n_loci}  and DATA_TYPE=={msa_type}" '
    logging.info(f'about to run {cmd_raxml_grove} to simulate tree')
    execute_command_and_write_to_log(cmd_raxml_grove)
    logging.info("Done with simulations")
    folder = os.listdir(out_dir)[0]
    true_tree_path = os.path.join(out_dir, folder, 'tree_best.newick')
    model_formulation_file = os.path.join(out_dir, folder, 'tree_best.newick.log')
    model = extract_model_specification_from_log(model_formulation_file, 'model')
    length = extract_model_specification_from_log(model_formulation_file, 'length')
    sim_dict = {'model': model, 'model_short': re.sub(r'\{[^{}]*\}','',model) , 'length': length, 'tree_id': folder,'tree_folder': os.path.join(out_dir, folder), 'tree_id': folder,'true_tree_path': true_tree_path}
    return sim_dict








def single_simulated_MSA_pipeline(tree_sim_dict, curr_run_dir, args):
    msa_path = simulate_msa(output_prefix=os.path.join(curr_run_dir, 'sim_msa'), tree_file=tree_sim_dict["true_tree_path"],
                            model=tree_sim_dict["model"], length=tree_sim_dict["length"])
    logging.info("Evaluating likelihood on current MSA")
    msa_results_dict = get_msa_stats(msa_path, tree_sim_dict["model"])
    msa_results_dict.update(tree_sim_dict)
    tree_searches_folder = os.path.join(curr_run_dir, "all_tree_searches")
    results_folder = os.path.join(curr_run_dir, "results_folder")
    create_or_clean_dir(tree_searches_folder)
    create_or_clean_dir(results_folder)
    boot_tree_raxml =  raxml_bootstrap_pipeline(tree_searches_folder,results_folder , msa_path, prefix ="boot", model =tree_sim_dict["model_short"],  n_cpus=1,
                                                n_workers='auto')
    boot_tree_iqtree = iqtree_pipeline(tree_searches_folder,results_folder , msa_path, model = tree_sim_dict["model_short"], nb=args.nb_iqtree, prefix = "iqtree_boot")
    boot_tree_fasttree = fasttree_pipeline(tree_searches_folder,results_folder ,msa_path, msa_type = args.msa_type, nb = args.nb_fasttree)
    for d in [boot_tree_raxml, boot_tree_iqtree, boot_tree_fasttree]:
        d.update(msa_results_dict)
        d.update(tree_sim_dict)
        d['msa_path'] = msa_path
    #create_or_clean_dir(tree_searches_folder) # remove redundant files
    return boot_tree_raxml, boot_tree_iqtree,boot_tree_fasttree


def add_features_to_data(msa_dir,boot_tree_raxml_metrics,boot_tree_iqtree_metrics,boot_tree_fasttree_metrics,tree_sim_dict):
        curr_boot_tree_raxml_df, raxml_feature_running_time = msa_path_bootstrap_analysis(msa_dir, mle_path=
        boot_tree_raxml_metrics['final_tree_topology_path'], true_tree_path=tree_sim_dict['true_tree_path'],
                                                                                          program='raxml',
                                                                                          bootstrap_tree_details_dict=boot_tree_raxml_metrics,
                                                                                          n_pars=100)
        curr_boot_tree_raxml_df["feature_running_time"] = raxml_feature_running_time
        curr_boot_tree_iqtree_df, iqtree_feature_running_time = msa_path_bootstrap_analysis(msa_dir, mle_path=
        boot_tree_iqtree_metrics['final_tree_ultrafast'], true_tree_path=tree_sim_dict['true_tree_path'],
                                                                                            program='iqtree',
                                                                                            bootstrap_tree_details_dict=boot_tree_iqtree_metrics,
                                                                                            n_pars=100)
        curr_boot_tree_iqtree_df["feature_running_time"] = iqtree_feature_running_time
        curr_boot_tree_fasttree_df, fasttree_feature_running_time = msa_path_bootstrap_analysis(msa_dir, mle_path=
        boot_tree_fasttree_metrics['sh_bootstrap'], true_tree_path=tree_sim_dict['true_tree_path'],
                                                                                                program='fasttree',
                                                                                                bootstrap_tree_details_dict=boot_tree_fasttree_metrics,
                                                                                                n_pars=100)
        curr_boot_tree_fasttree_df["feature_running_time"] = fasttree_feature_running_time
        return curr_boot_tree_raxml_df, curr_boot_tree_iqtree_df, curr_boot_tree_fasttree_df

def main():
    parser = job_parser()
    args = parser.parse_args()
    curr_job_folder = args.curr_job_folder
    create_dir_if_not_exists(curr_job_folder)
    curr_job_general_log_file = os.path.join(args.curr_job_folder, "log_file.log")
    logging.basicConfig(filename=curr_job_general_log_file, level=logging.INFO)
    all_results_df_raxml = pd.DataFrame()
    all_results_df_iqtree = pd.DataFrame()
    all_results_df_fasttree = pd.DataFrame()
    for i in range(args.number_of_trees):
        logging.info(f"Starting simulation of tree {i}")
        #try:
        logging.info(f"Simulating tree {i} in {curr_job_folder}")
        curr_tree_folder = os.path.join(curr_job_folder, f'raxml_tree_{i}')
        tree_sim_dict = RAxML_grove_tree_simulation(curr_tree_folder, args.min_n_taxa,
                                                                         args.max_n_taxa, args.min_n_loci,
                                                                         args.max_n_loci, args.msa_type)
        for j in range(args.number_of_MSAs_per_tree):
            logging.info(f"Starting with MSA {j} ")
            msa_dir = os.path.join(tree_sim_dict["tree_folder"], f"iqtree_msa_{j}")
            create_dir_if_not_exists(msa_dir)
            logging.info(f"Simulating MSA {j} in {msa_dir}")
            boot_tree_raxml_metrics, boot_tree_iqtree_metrics,boot_tree_fasttree_metrics = single_simulated_MSA_pipeline(tree_sim_dict, msa_dir, args)
            if args.calc_features:
                try:
                    curr_boot_tree_raxml_df, curr_boot_tree_iqtree_df, curr_boot_tree_fasttree_df = add_features_to_data(msa_dir,boot_tree_raxml_metrics,boot_tree_iqtree_metrics,boot_tree_fasttree_metrics,tree_sim_dict)
                except:
                    logging.error("Could not calculate features on current msa")
                    continue
            else:
                curr_boot_tree_raxml_df = pd.DataFrame([boot_tree_raxml_metrics])
                curr_boot_tree_iqtree_df = pd.DataFrame([boot_tree_raxml_metrics])
                curr_boot_tree_fasttree_df = pd.DataFrame([boot_tree_raxml_metrics])
        #### Update global dataframes ####
            all_results_df_raxml = pd.concat([curr_boot_tree_raxml_df,all_results_df_raxml])
            all_results_df_iqtree = pd.concat([curr_boot_tree_iqtree_df, all_results_df_iqtree])
            all_results_df_fasttree = pd.concat([curr_boot_tree_fasttree_df, all_results_df_fasttree])

        ### Write temporary files to csv ###
        all_results_df_raxml.to_csv(os.path.join(curr_job_folder, "simulations_df_raxml.tsv"), sep='\t')
        all_results_df_iqtree.to_csv(os.path.join(curr_job_folder, "simulations_df_iqtree.tsv"), sep='\t')
        all_results_df_fasttree.to_csv(os.path.join(curr_job_folder, "simulations_df_fasttree.tsv"), sep='\t')


if __name__ == "__main__":
    main()
