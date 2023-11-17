import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from side_code.config import *
from side_code.code_submission import execute_command_and_write_to_log
from programs.raxml import raxml_bootstrap_pipeline, remove_redundant_sequences
from programs.FastTree import fasttree_pipeline
from programs.IQTREE import iqtree_pipeline
from simulations_generation.simulations_argparser import  job_parser
from simulations_generation.msa_features import get_msa_stats
from side_code.MSA_manipulation import get_MSA_seq_names
from simulation_edit.bootstrap_edit import msa_path_edit_analysis
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir
from programs.raxml_grove import RAxML_grove_tree_simulation
import pandas as pd
import os
from ete3 import Tree


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


def obtain_tree_search_bootstrap_results(msa_path, tree_sim_dict, curr_run_dir, tree_search_model, args):
    logging.info("Evaluating likelihood on current MSA")
    msa_results_dict = get_msa_stats(msa_path, tree_search_model)
    msa_results_dict.update(tree_sim_dict)
    tree_searches_folder = os.path.join(curr_run_dir, "all_tree_searches")
    results_folder = os.path.join(curr_run_dir, "results_folder")
    create_or_clean_dir(tree_searches_folder)
    create_or_clean_dir(results_folder)
    logging.info(f"Model mode is {tree_sim_dict['model_mode']}, using {tree_search_model} model for tree search")
    boot_tree_fasttree = fasttree_pipeline(tree_searches_folder, results_folder, msa_path, msa_type=args.msa_type,
                                           nb=args.nb_fasttree, model = tree_search_model)
    boot_tree_raxml =  raxml_bootstrap_pipeline(tree_searches_folder,results_folder , msa_path, prefix ="boot", model =tree_search_model,  n_cpus=1,
                                                n_workers='auto')
    boot_tree_iqtree = iqtree_pipeline(tree_searches_folder,results_folder , msa_path, model = tree_search_model, nb=args.nb_iqtree, prefix = "iqtree_boot")
    for d in [boot_tree_raxml, boot_tree_iqtree, boot_tree_fasttree]:
        d.update(msa_results_dict)
        d.update(tree_sim_dict)
        d['msa_path'] = msa_path
        d['tree_search_model'] = tree_search_model
    #create_or_clean_dir(tree_searches_folder) # remove redundant files
    return boot_tree_raxml, boot_tree_iqtree,boot_tree_fasttree


def add_features_to_data(msa_path,msa_dir,boot_tree_raxml_metrics,boot_tree_iqtree_metrics,boot_tree_fasttree_metrics,tree_sim_dict):
        curr_boot_tree_raxml_df = msa_path_edit_analysis(msa_path, msa_dir, mle_path=
        boot_tree_raxml_metrics['final_tree_topology_path'], true_tree_path=tree_sim_dict['true_tree_path'],
                                                                                                                 program='raxml',
                                                                                                                 bootstrap_tree_details_dict=boot_tree_raxml_metrics,
                                                                                                                 n_pars=100)

        curr_boot_tree_iqtree_df = msa_path_edit_analysis(msa_path, msa_dir, mle_path=
        boot_tree_iqtree_metrics['final_tree_ultrafast'], true_tree_path=tree_sim_dict['true_tree_path'],
                                                                                                                   program='iqtree',
                                                                                                                   bootstrap_tree_details_dict=boot_tree_iqtree_metrics,
                                                                                                                   n_pars=100)
        curr_boot_tree_fasttree_df = msa_path_edit_analysis(msa_path, msa_dir, mle_path=
        boot_tree_fasttree_metrics['sh_bootstrap'], true_tree_path=tree_sim_dict['true_tree_path'],
                                                                                                                         program='fasttree',
                                                                                                                         bootstrap_tree_details_dict=boot_tree_fasttree_metrics,
                                                                                                                         n_pars=100)
        return curr_boot_tree_raxml_df, curr_boot_tree_iqtree_df, curr_boot_tree_fasttree_df


def obtain_tree_sim_dict(i, args,curr_job_folder, tree_id):
    print(f"i={i}")
    logging.info(f"Starting simulation of tree {i}")
    logging.info(f"Simulating tree {i} in {curr_job_folder}")
    curr_tree_folder = os.path.join(curr_job_folder, f'raxml_tree_{i}')
    create_dir_if_not_exists(curr_tree_folder)
    raxml_grove_simulations_folder = os.path.join(curr_tree_folder, "original_raxml_grove_files")
    create_dir_if_not_exists(raxml_grove_simulations_folder)

    all_models_tree_sim_dicts = RAxML_grove_tree_simulation(raxml_grove_simulations_folder, args.min_n_taxa,
                                                    args.max_n_taxa, args.min_n_loci,
                                                    args.max_n_loci, args.msa_type, tree_id, model_modes=args.model_modes.split('_'))

    return all_models_tree_sim_dicts


def obtain_all_results_per_MSA(tree_sim_dict, j, args):
    logging.info(f"Starting with MSA {j} ")
    msa_dir = os.path.join(tree_sim_dict["tree_folder"], f"iqtree_msa_{j}_{tree_sim_dict['model_mode']}")
    create_dir_if_not_exists(msa_dir)
    logging.info(f"Simulating MSA {j} in {msa_dir}")
    msa_path = simulate_msa(output_prefix=os.path.join(msa_dir, 'sim_msa'),
                            tree_file=tree_sim_dict["true_tree_path_orig"],
                            model=tree_sim_dict["simulation_model"], length=tree_sim_dict["length"])

    local_true_tree_path = os.path.join(msa_dir, "tree_best_pruned.newick")
    tree_sim_dict["true_tree_path"] = local_true_tree_path
    all_searches_pipeline_folder = os.path.join(msa_dir, "searches_pipeline")
    create_dir_if_not_exists(all_searches_pipeline_folder)
    if tree_sim_dict['model_mode']=='standard':
        tree_search_model = tree_sim_dict["simulation_model_short"]
    elif tree_sim_dict['model_mode']=='downgrade':
        tree_search_model = 'JC'
    else:
        tree_search_model = 'GTR+F+G+I'
    remove_redundant_sequences(msa_dir, prefix="check", msa_path=msa_path, model=tree_search_model)
    tree = Tree(tree_sim_dict["true_tree_path_orig"], format=1)
    tree.prune(get_MSA_seq_names(msa_path))
    tree.write(outfile=local_true_tree_path, format=1)
    boot_tree_raxml_metrics, boot_tree_iqtree_metrics, boot_tree_fasttree_metrics = obtain_tree_search_bootstrap_results(
        msa_path, tree_sim_dict, all_searches_pipeline_folder,tree_search_model, args)

    all_features_folder = os.path.join(msa_dir, "features_pipeline")
    create_dir_if_not_exists(all_features_folder)
    if args.calc_features:
        curr_boot_tree_raxml_df, curr_boot_tree_iqtree_df, curr_boot_tree_fasttree_df = add_features_to_data(msa_path,
                                                                                                             all_features_folder,
                                                                                                             boot_tree_raxml_metrics,
                                                                                                             boot_tree_iqtree_metrics,
                                                                                                             boot_tree_fasttree_metrics,
                                                                                                             tree_sim_dict)
    else:
        curr_boot_tree_raxml_df = pd.DataFrame([boot_tree_raxml_metrics])
        curr_boot_tree_iqtree_df = pd.DataFrame([boot_tree_iqtree_metrics])
        curr_boot_tree_fasttree_df = pd.DataFrame([boot_tree_fasttree_metrics])
    return curr_boot_tree_raxml_df,curr_boot_tree_iqtree_df,curr_boot_tree_fasttree_df


def obtain_all_results_per_tree(args, tree_sim_dict,curr_job_folder,all_results_df_raxml,all_results_df_iqtree,all_results_df_fasttree):

    for j in range(args.number_of_MSAs_per_tree):
            curr_boot_tree_raxml_df, curr_boot_tree_iqtree_df, curr_boot_tree_fasttree_df = obtain_all_results_per_MSA(
                tree_sim_dict, j,  args)
            all_results_df_raxml = pd.concat([curr_boot_tree_raxml_df, all_results_df_raxml])
            all_results_df_iqtree = pd.concat([curr_boot_tree_iqtree_df, all_results_df_iqtree])
            all_results_df_fasttree = pd.concat([curr_boot_tree_fasttree_df, all_results_df_fasttree])

    all_results_df_raxml.to_csv(os.path.join(curr_job_folder, "simulations_df_raxml.tsv"), sep='\t')
    all_results_df_iqtree.to_csv(os.path.join(curr_job_folder, "simulations_df_iqtree.tsv"), sep='\t')
    all_results_df_fasttree.to_csv(os.path.join(curr_job_folder, "simulations_df_fasttree.tsv"), sep='\t')
    return all_results_df_raxml,all_results_df_iqtree,all_results_df_fasttree

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
    if args.pre_chosen_tree_ids:
        logging.info("Using pre chosen tree IDs")
        with open(args.cur_job_trees_file) as JOB_TREES_FILE:
            tree_ids = JOB_TREES_FILE.read().split("\n")
        for i,tree_id in enumerate(tree_ids):
            logging.info(f"Tree id = {tree_id}")
            try:
                all_models_tree_sim_dict = obtain_tree_sim_dict(i, args, curr_job_folder, tree_id)
                for model_mode in all_models_tree_sim_dict:
                    all_results_df_raxml,all_results_df_iqtree,all_results_df_fasttree = obtain_all_results_per_tree(args, all_models_tree_sim_dict[model_mode], curr_job_folder, all_results_df_raxml,
                                                all_results_df_iqtree, all_results_df_fasttree)
                logging.info("Done with current tree id")
                break
            except:
                logging.info(f"Could not run on id {tree_id}, trying next")
    else:
        logging.info("Using random chosen tree IDs")
        for i in range(args.number_of_trees):
            all_models_tree_sim_dict = obtain_tree_sim_dict(i, args, curr_job_folder, tree_id = None)
            for model_mode in all_models_tree_sim_dict:
                all_results_df_raxml,all_results_df_iqtree,all_results_df_fasttree = obtain_all_results_per_tree(args, all_models_tree_sim_dict[model_mode], curr_job_folder, all_results_df_raxml,
                                            all_results_df_iqtree, all_results_df_fasttree)




if __name__ == "__main__":
    main()
