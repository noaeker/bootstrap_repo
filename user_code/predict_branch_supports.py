from simulation_edit.feature_extraction import extract_all_features_per_mle
import argparse
import os
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
import pickle
import shutil
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--mle_tree_path', type=str, default="/Users/noa/Workspace/bootstrap_results/test4/job_0/raxml_tree_0/original_raxml_grove_files/9130/iqtree_msa_0_upgrade/searches_pipeline/results_folder/raxml_final_tree_topology.tree")
    parser.add_argument('--all_mles_tree_path', type=str, default='/Users/noa/Workspace/bootstrap_results/test4/job_0/raxml_tree_0/original_raxml_grove_files/9130/iqtree_msa_0_upgrade/searches_pipeline/results_folder/raxml_all_final_tree_topologies.tree')
    parser.add_argument('--msa_path', type=str, default='/Users/noa/Workspace/bootstrap_results/test4/job_0/raxml_tree_0/original_raxml_grove_files/9130/iqtree_msa_0_upgrade/sim_msa.fa')
    parser.add_argument('--model', type=str, default="GTR+G")
    parser.add_argument('--trained_ML_model_path', type=str, default="/Users/noa/Workspace/bootstrap_results/remote_results/ML_results/raxml/final_model/full_model/model_stadard")
    args = parser.parse_args()
    tmp_files_fodler = os.path.join(args.working_dir, 'tmp_feature_files')
    create_dir_if_not_exists(tmp_files_fodler)
    final_tree,features_df = extract_all_features_per_mle(tmp_files_fodler,args.msa_path,args.model, args.mle_tree_path, {}, args.all_mles_tree_path)
    model = pickle.load(open(args.trained_ML_model_path, 'rb'))
    shutil.rmtree(tmp_files_fodler)
    feature_names = model['best_model'].feature_name_
    features_df['predicted_bootstrap_score'] = model['best_model'].predict_proba(features_df[feature_names])[:, 1]
    for i,row in features_df.iterrows():
        (final_tree&row["node_name"]).support = row["predicted_bootstrap_score"]
    final_tree.write(format=0, outfile="tree_ML_bp.nw")

if __name__ == "__main__":
    main()