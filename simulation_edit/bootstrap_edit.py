import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from simulation_edit.simulations_edit_argparser import job_parser
from simulation_edit.feature_extraction import extract_all_features_per_mle
from side_code.file_handling import *

def get_program_default_ML_tree(program):
    if program =='raxml':
        name = 'final_tree_topology_path'
    elif program =='iqtree':
        name = 'final_tree_ultrafast'
    elif program=='fasttree':
        name = 'sh_bootstrap'
    return name

def get_bootstrap_and_all_mles_path(program, bootstrap_tree_details):
    all_mles_path = None
    extra_support_values= {}
    if program == 'raxml':
        all_mles_path = bootstrap_tree_details['all_final_tree_topologies_path']
    elif program == 'iqtree':
        final_tree_aLRT = bootstrap_tree_details['final_tree_aLRT']
        final_tree_aBayes_path = bootstrap_tree_details['final_tree_aBayes']
        extra_support_values = {'aLRT_iqtree': final_tree_aLRT, 'aBayes_iqtree': final_tree_aBayes_path}
    return extra_support_values, all_mles_path



def main():
    parser = job_parser()
    args = parser.parse_args()
    create_dir_if_not_exists(args.job_work_path)
    data = pd.read_csv(args.job_data_path, sep='\t')
    all_splits = pd.DataFrame()
    log_file_path = os.path.join(args.job_work_path, "general_features.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logging.info(f"Number of different trees is {len(data['true_tree_path'].unique())}")
    for i,true_tree_path in enumerate(data['true_tree_path'].unique()):
        logging.info(f"Tree {i} out of {len(data['true_tree_path'].unique())}")
        tree_data = data.loc[data.true_tree_path == true_tree_path]
        for program in tree_data['program'].unique():
            logging.info(f"Program = {program}")
            tree_program_data = tree_data.loc[tree_data.program==program]
            for j,msa_path in enumerate(tree_data['msa_path'].unique()):
                working_dir = os.path.join(args.job_work_path,f"msa_{j}")
                create_dir_if_not_exists(working_dir)
                logging.info(f"MSA {j} out of {len(tree_data['msa_path'].unique())}")
                bootstrap_tree_details = tree_program_data.loc[tree_program_data.msa_path == msa_path].head(1).squeeze()

                mle_tree_path =  bootstrap_tree_details[get_program_default_ML_tree(program)]
                model = bootstrap_tree_details["tree_search_model"]
                program_bootstrap_support_paths, all_mle_path = get_bootstrap_and_all_mles_path(program, bootstrap_tree_details)
                tree_obj_with_features, all_msa_splits_df = extract_all_features_per_mle(working_dir, msa_path, model, mle_tree_path, extra_bootstrap_support_paths =program_bootstrap_support_paths , all_mles_tree_path =all_mle_path, true_tree_path = true_tree_path)
                all_msa_splits_df["program"] = program
                all_msa_splits_df = pd.concat([all_msa_splits_df,bootstrap_tree_details],axis=1)

                all_splits = pd.concat([all_splits, all_msa_splits_df])
                all_splits.to_csv(args.job_final_output_path, sep='\t')
        raxml_data = all_splits.loc[all_splits.program=='raxml']
        raxml_data.to_csv(os.path.join(args.job_work_path,'simulations_df_raxml.tsv'), sep='\t')
        iqtree_data = all_splits.loc[all_splits.program == 'iqtree']
        iqtree_data.to_csv(os.path.join(args.job_work_path,'simulations_df_iqtree.tsv'), sep='\t')
        fasttree_data = all_splits.loc[all_splits.program == 'fasttree']
        fasttree_data.to_csv(os.path.join(args.job_work_path,'simulations_df_fasttree.tsv'), sep='\t')
    logging.info("Done with all trees!")




if __name__ == "__main__":
    main()