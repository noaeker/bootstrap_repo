
import argparse
import pandas as pd
import logging
import os


def unify_csvs(csvs_path_list, unified_csv_path):
    tmp_dataframes = []
    for csv_path in csvs_path_list:
        tmp_df = pd.read_csv(csv_path,sep='\t')
        tmp_dataframes.append(tmp_df)
    combined_df = pd.concat(tmp_dataframes, sort=False)
    combined_df.to_csv(unified_csv_path,sep = '\t')
    return combined_df


def unify_results_across_jobs(raw_results_folder, name,n_jobs = 1000):
    out_csv_path = os.path.join(raw_results_folder, f'{name}.tsv')
    if os.path.exists(out_csv_path):
        logging.info(f"Using existing data in {out_csv_path}")
        return out_csv_path
    logging.info("unifying csvs from all_jobs")
    csv_paths = [f"{raw_results_folder}/job_{i}/{name}.tsv" for i in range(n_jobs)]
    existing_csv_path = [path for path in csv_paths if os.path.exists(path)]
    unify_csvs(existing_csv_path, out_csv_path)
    logging.info(f"Csvs were unified to {out_csv_path}")
    return out_csv_path


def main():
    out_path = unify_results_across_jobs(raw_results_folder='/Users/noa/Workspace/bootstrap_results/bootstrap_edit_results', name = 'job_raw_data_with_features')
    data = pd.read_csv(out_path, sep = '\t')
    pass


if __name__ == "__main__":
    main()
