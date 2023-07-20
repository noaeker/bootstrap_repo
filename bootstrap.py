from side_code.raxml import raxml_bootstrap_search
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir
import os





def main():
    base_folder = '/Users/noa/Workspace/bootstrap_results'
    curr_run_directory = os.path.join(base_folder, "test")
    create_or_clean_dir(curr_run_directory)
    n_bootstrap_replicates = 10
    msa_path = '/Users/noa/Workspace/simulations_results/raxml_grove_simulations/job_0/raxml_tree_0/52454/assembled_sequences.fasta'
    prefix = 'boot'
    model = 'GTR+G'
    bootstrap_results = raxml_bootstrap_search(curr_run_directory, msa_path, prefix, model, n_bootstrap_replicates, n_cpus=1,
                           n_workers='auto')
    print(bootstrap_results)



if __name__ == "__main__":
    main()
