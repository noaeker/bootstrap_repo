import sys
import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from side_code.config import *
from side_code.basic_trees_manipulation import get_tree_string
from side_code.code_submission import execute_command_and_write_to_log
from side_code.raxml import raxml_bootstrap_search
from simulations_generation.simulations_argparser import  job_parser
from simulations_generation.msa_features import get_msa_stats
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir
import sys
import pandas as pd
import os
import re



def simulate_msa(output_prefix, tree_file, model, length, ret_cmd = False):
    '''

    :param output_prefix:
    :param model: WAG / GTR
    :param treefile:
    :return:
    '''
    command = f"{IQTREE_SIM_PATH} --alisim {output_prefix} -m {model} -t {tree_file}  --length {int(length)}" \
        f"  "
    # logging.info(f"About to run {command}")
    execute_command_and_write_to_log(command)
    msa_path = f"{output_prefix}.phy"
    print(command)
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




def RAxML_grove_tree_simulation( out_dir, min_n_taxa, max_n_taxa, min_n_loci, max_n_loci):
    cmd_raxml_grove = f'{RAxML_alisim_path} generate -g alisim -o {out_dir} -q " NUM_TAXA>={min_n_taxa} and NUM_TAXA<={max_n_taxa} and OVERALL_NUM_ALIGNMENT_SITES>={min_n_loci} and OVERALL_NUM_ALIGNMENT_SITES<={max_n_loci} and OVERALL_NUM_PARTITIONS==1" '
    logging.info(f'about to run {cmd_raxml_grove}')
    execute_command_and_write_to_log(cmd_raxml_grove)
    folder = os.listdir(out_dir)[0]
    tree_path = os.path.join(out_dir, folder, 'tree_best.newick')
    model_formulation_file = os.path.join(out_dir, folder, 'tree_best.newick.log')
    model = extract_model_specification_from_log(model_formulation_file, 'model')
    length = extract_model_specification_from_log(model_formulation_file, 'length')
    sim_dict = {'model': model, 'model_short': re.sub(r'\{[^{}]*\}','',model) , 'length': length, 'tree_id': folder,'tree_folder': os.path.join(out_dir, folder), 'tree_id': folder,'tree_path': tree_path, 'tree_str': get_tree_string(tree_path)}
    return sim_dict








def single_simulated_MSA_pipeline(tree_sim_dict, j, args):
    iqtree_folder = os.path.join(tree_sim_dict["tree_folder"], f"iqtree_msa_{j}")
    create_dir_if_not_exists(iqtree_folder)
    logging.info(f"Simulating MSA {j} in {iqtree_folder}")
    msa_path = simulate_msa(output_prefix=os.path.join(iqtree_folder, 'sim_msa'), tree_file=tree_sim_dict["tree_path"],
                            model=tree_sim_dict["model"], length=tree_sim_dict["length"])
    logging.info("Evaluating likelihood on current MSA")
    msa_results_dict = get_msa_stats(msa_path, tree_sim_dict["model"])
    msa_results_dict.update(tree_sim_dict)
    tree_searches_folder = os.path.join(iqtree_folder, "all_tree_searches")
    create_or_clean_dir(tree_searches_folder)
    bootstrap_tree_search =  raxml_bootstrap_search(tree_searches_folder, msa_path, prefix = "boot", model =tree_sim_dict["model_short"] , n_bootstrap_replicates = args.bs, n_cpus=1,
                           n_workers='auto')
    print(bootstrap_tree_search)
    bootstrap_tree_search.update(msa_results_dict)
    #create_or_clean_dir(tree_searches_folder)
    return bootstrap_tree_search



def main():
    parser = job_parser()
    args = parser.parse_args()
    curr_job_folder = args.curr_job_folder
    create_or_clean_dir(curr_job_folder)
    curr_job_general_log_file = os.path.join(args.curr_job_folder, "log_file.log")
    logging.basicConfig(filename=curr_job_general_log_file, level=logging.INFO)
    all_results_df = pd.DataFrame()
    for i in range(args.number_of_trees):
        logging.info(f"Starting simulation of tree {i}")
        #try:
        logging.info(f"Simulating tree {i} in {curr_job_folder}")
        curr_tree_folder = os.path.join(curr_job_folder, f'raxml_tree_{i}')
        tree_sim_dict = RAxML_grove_tree_simulation(curr_tree_folder, args.min_n_taxa,
                                                                         args.max_n_taxa, args.min_n_loci,
                                                                         args.max_n_loci)
        for j in range(args.number_of_MSAs_per_tree):
            bootstrap_reslt = single_simulated_MSA_pipeline(tree_sim_dict, j, args,
                                                           )
            all_results_df = all_results_df.append(bootstrap_reslt, ignore_index=True)
       #except Exception as E:
       #         logging.error(f"Could not run simulation at iteration {i}\n Exception : {E}")
        logging.info(f"Updating tree {i} to csv")
        all_results_df.to_csv(os.path.join(curr_job_folder, "simulations_df.tsv"), sep='\t')


if __name__ == "__main__":
    main()
