
import argparse
import pandas as pd
import os


def unify_csvs(csvs_path_list, unified_csv_path):
    tmp_dataframes = []
    for csv_path in csvs_path_list:
        tmp_df = pd.read_csv(csv_path,sep='\t')
        print("size = "+ str(len(tmp_df.index)))
        tmp_dataframes.append(tmp_df)
    combined_df = pd.concat(tmp_dataframes, sort=False)
    combined_df.to_csv(unified_csv_path,sep = '\t')
    return combined_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_results_folder', action='store', type=str)
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--n_jobs',action='store', type = int, default= 10)
    args = parser.parse_args()
    print(f'Job arguments are: {args}')
    csv_paths= [f"{args.raw_results_folder}/job_{i}/{args.name}.tsv" for i in range(args.n_jobs)]
    existing_csv_path = [path for path in csv_paths if os.path.exists(path)]
    out_csv_path = os.path.join(args.raw_results_folder,f'{args.name}.tsv')
    unify_csvs(existing_csv_path, out_csv_path)

if __name__ == "__main__":
    main()