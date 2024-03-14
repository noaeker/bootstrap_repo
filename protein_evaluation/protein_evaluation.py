import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)
import argparse


from side_code.config import *
from side_code.code_submission import execute_command_and_write_to_log
from programs.raxml import raxml_no_bootstrap_pipeline, remove_redundant_sequences
from programs.FastTree import fasttree_pipeline
from programs.IQTREE import iqtree_pipeline
from simulations_generation.simulations_argparser import  job_parser
from simulation_edit.msa_features import get_msa_stats
from side_code.MSA_manipulation import get_MSA_seq_names
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir
from simulation_edit.feature_extraction import extract_all_features_per_mle
from programs.raxml_grove import RAxML_grove_tree_simulation
import pandas as pd
import os
from ete3 import Tree
import pickle
import random
import math




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_results_folder', action='store', type=str, default = "/Users/noa/Workspace/bootstrap_results/protein_eval")
    parser.add_argument('--name', action='store', type=str, default = 'protein_eval')
    parser.add_argument('--nni_model_path', action='store', type=str, default="/Users/noa/Workspace/bootstrap_results/remote_results/ML_results/raxml/final_model/full_model/model_stadard")
    parser.add_argument('--no_nni_model_path', action='store', type=str, default="/Users/noa/Workspace/bootstrap_results/remote_results/ML_results/raxml/final_model/fast_model/model_stadard")
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--n_MSAs', type=int, default=100)
    parser.add_argument('--min_taxa', type=int, default=8)
    parser.add_argument('--max_taxa', type=int, default=10 )
    parser.add_argument('--min_loci', type=int, default=30)
    parser.add_argument('--max_loci', type=int, default=60)
    args = parser.parse_args()
    wd = args.raw_results_folder
    create_or_clean_dir(wd)
    all_features_df = pd.DataFrame()
    out_csv_path = os.path.join(wd, 'protein_prediction.csv')
    log_file_path = os.path.join(wd, 'log_file.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO)

    for i in range(args.n_MSAs):
        try:
            results_folder = os.path.join(wd,'results')
            create_or_clean_dir(results_folder)
            iqtree_sim_folder = os.path.join(wd,'iqtree_alisim')
            create_or_clean_dir(iqtree_sim_folder)
            iqtree_sim_path = os.path.join(iqtree_sim_folder,args.name)
            n_taxa = random.randint(args.min_taxa,args.max_taxa)#(30, 200)
            n_loci = random.randint(args.min_loci,args.max_loci)#(100,20000)
            logging.info(f"MSA {i}, n_taxa = {n_taxa}, n_loci = {n_loci}")
            cmd = f'{IQTREE_SIM_PATH} --alisim {iqtree_sim_path} --out-format fasta -t RANDOM{{yh/{n_taxa}}} --length {n_loci} -m WAG+G'
            execute_command_and_write_to_log(cmd)
            msa_path = os.path.join(iqtree_sim_path+".fa")
            true_tree_path = os.path.join(iqtree_sim_path+".treefile")
            tmp_dir = os.path.join(wd,'tmp')
            create_dir_if_not_exists(tmp_dir)
            remove_redundant_sequences(tmp_dir, prefix="check", msa_path=msa_path, model='WAG+G')
            tree = Tree(true_tree_path, format=1)
            tree.prune(get_MSA_seq_names(msa_path))
            local_true_tree_path = os.path.join(wd, "tree_best_pruned.newick")
            tree.write(outfile=local_true_tree_path, format=1)
            raxml_results = os.path.join(wd,'raxml_results')
            create_or_clean_dir(raxml_results)
            raxml_boot = raxml_no_bootstrap_pipeline(raxml_results, results_folder, msa_path, prefix = "raxml_boot", model = 'WAG+G', n_cpus = args.n_cpus, n_workers ='auto', use_existing_trees = False)
            msa_results_dict = get_msa_stats(msa_path, 'WAG+G')
            raxml_boot.update(msa_results_dict)
            raxml_boot['msa_path'] = msa_path
            raxml_boot['tree_search_model'] = 'WAG+G'
            feature_extraction_dir = os.path.join(wd,'feautres_calc')
            create_or_clean_dir(feature_extraction_dir)
            final_tree,features = extract_all_features_per_mle(feature_extraction_dir, msa_path, model = 'WAG+G', mle_tree_path = raxml_boot['final_tree_topology_path'], extra_bootstrap_support_paths = {}, all_mles_tree_path = raxml_boot['all_final_tree_topologies_path'], true_tree_path = local_true_tree_path, booster_program_path = None, raxml_program_path = None, mad_program_path = None,n_cpus  = min(args.n_cpus,math.ceil(n_loci/800)))
            #features = features.apply(lambda row: row.map(raxml_boot), axis=1)
            nni_model = pickle.load(open(args.nni_model_path, 'rb'))
            feature_names_nni = nni_model['best_model'].feature_name_
            features['predicted_bootstrap_score_nni'] = nni_model['calibrated_model'].predict_proba(
                features[feature_names_nni])[:, 1]
            no_nni_model = pickle.load(open(args.no_nni_model_path, 'rb'))
            feature_names_no_nni = no_nni_model['best_model'].feature_name_
            features['predicted_bootstrap_score_no_nni'] = no_nni_model['calibrated_model'].predict_proba(
                features[feature_names_no_nni])[:, 1]
            features["msa_ind"] = i
            all_features_df = pd.concat([all_features_df,features])
            all_features_df.to_csv(out_csv_path)
        except:
            logging.info(f"Could not run on MSA {i}")
            pass



if __name__ == "__main__":
    main()
