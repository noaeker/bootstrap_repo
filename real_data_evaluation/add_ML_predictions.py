import argparse
import pickle
import pandas as pd
import os
from ete3 import Tree


def unify_csvs(csvs_path_list):
    tmp_dataframes = []
    for csv_path in csvs_path_list:
        try:
            tmp_df = pd.read_csv(csv_path)
            tmp_df['csv_path'] = csv_path
            print("size = "+ str(len(tmp_df.index)))
            tmp_dataframes.append(tmp_df)
        except:
            print("Could not run on current csv path")
    combined_df = pd.concat(tmp_dataframes, sort=False)
    return combined_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_results_folder', action='store', type=str, default = "/Users/noa/Workspace/bootstrap_results/real_data_eval")
    parser.add_argument('--name', action='store', type=str, default = 'raxml_real_data')
    parser.add_argument('--n_jobs',action='store', type = int, default= 1)
    parser.add_argument('--nni_model_path', action='store', type=str, default="/Users/noa/Workspace/bootstrap_results/remote_results/ML_results/raxml/final_model/full_model/model_stadard")
    parser.add_argument('--no_nni_model_path', action='store', type=str, default="/Users/noa/Workspace/bootstrap_results/remote_results/ML_results/raxml/final_model/fast_model/model_stadard")
    parser.add_argument('--ML_tree_path',action='store', type=str, default = "/Users/noa/Workspace/bootstrap_results/real_data_eval/job_0/new_tree.nw")
    args = parser.parse_args()
    print(f'Job arguments are: {args}')
    csv_paths= [f"{args.raw_results_folder}/job_{i}/{args.name}.csv" for i in range(args.n_jobs)]
    existing_csv_path = [path for path in csv_paths if os.path.exists(path)]
    out_csv_path = os.path.join(args.raw_results_folder,f'{args.name}.csv')
    combined_df = unify_csvs(existing_csv_path)
    #ML_tree = Tree(args.ML_tree_path, format = 1)
    nni_model = pickle.load(open(args.nni_model_path, 'rb'))
    feature_names_nni = nni_model['best_model'].feature_name_
    combined_df['predicted_bootstrap_score_nni'] = nni_model['calibrated_model'].predict_proba(combined_df[feature_names_nni])[:, 1]
    no_nni_model = pickle.load(open(args.no_nni_model_path, 'rb'))
    feature_names_no_nni = no_nni_model['best_model'].feature_name_
    combined_df['predicted_bootstrap_score_no_nni'] = no_nni_model['calibrated_model'].predict_proba(combined_df[feature_names_no_nni])[:, 1]
    combined_df.to_csv(out_csv_path)
    #for i, row in combined_df.iterrows():
    #    (ML_tree & row["node_name"]).support = row["predicted_bootstrap_score_nni"]
    #ML_tree.write(format=0, outfile=os.path.join(args.raw_results_folder, "ML_tree_with_predictions.nw"))


if __name__ == "__main__":
    main()
