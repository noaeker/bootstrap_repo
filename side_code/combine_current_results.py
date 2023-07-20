from side_code.config import *
import argparse
import pandas as pd



def unify_csvs(csvs_path_list, unified_csv_path):
    tmp_dataframes = []
    for csv_path in csvs_path_list:
        tmp_df = pd.read_csv(csv_path,sep=CSV_SEP)
        print("size = "+ str(len(tmp_df.index)))
        tmp_dataframes.append(tmp_df)
    combined_df = pd.concat(tmp_dataframes, sort=False)
    combined_df.to_csv(unified_csv_path,sep = CSV_SEP)
    return combined_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_results_folder', action='store', type=str, default = TMP_RESULTS_FOLDER)
    parser.add_argument('--n_jobs',action='store', type = int, default= 10)
    parser.add_argument('--out_csv_path', action='store', type=str, default=f'{RESULTS_FOLDER}/full_raxml_data{CSV_SUFFIX}')
    args = parser.parse_args()
    print(f'Job arguments are: {args}')
    csv_paths= [f"{args.raw_results_folder}/job_{i}/{i}.tsv" for i in range(args.n_jobs)]
    print(csv_paths)
    unify_csvs(csv_paths, args.out_csv_path)

if __name__ == "__main__":
    main()