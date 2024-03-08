import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from programs.FastTree import fasttree_pipeline
from programs.raxml import raxml_bootstrap_pipeline
from programs.IQTREE import iqtree_pipeline
from side_code.file_handling import create_dir_if_not_exists
from simulation_edit.bootstrap_edit import get_bootstrap_and_all_mles_path,get_splits_df
from real_data_evaluation.real_data_evaluation_parsers import job_parser
import os
import tarfile
import logging
import pickle

def convert_nexus_to_fasta(nexus_path, out_file):
    with open(nexus_path) as N:
        text = N.read()
        try:
            matrix_location = text.index('matrix')
        except:
            matrix_location = text.index('Matrix')
        end_location = text.index(';\nend;')
        text = text[matrix_location+6:end_location]
        lines = text.split('\n\t')

        sequences_dict = {}
        for seq in lines:
            if len(seq)>50:
                try:
                    seq_name_and_value = seq.split(' ')
                    sequences_dict[seq_name_and_value[0]] = seq_name_and_value[1]
                except:
                    seq_name_and_value = seq.split('\t')
                    sequences_dict[seq_name_and_value[0]] = seq_name_and_value[1]

        with open(out_file,'w') as Out:
            for seq in sequences_dict:
                if len(sequences_dict[seq].replace('-', '').replace(' ','').replace('\n','').replace('\t','')) > 0:
                    Out.write(">")
                    Out.write(seq+"\n")
                    seq = sequences_dict[seq]
                    Out.write(seq)
                    Out.write('\n')



#msa_path,model, true_tree_path, bootstrap_tree_details, program, working_dir


def run_programs(msa_path,tree_searches_folder, results_folder,msa_type, model, n_cpus, models_dict):
    tmp_files_folder = os.path.join(tree_searches_folder, 'tmp')
    create_dir_if_not_exists(tmp_files_folder)
    logging.info("Running IQTREE")
    boot_tree_iqtree_details = iqtree_pipeline(tree_searches_folder, results_folder, msa_path, model=model, nb=1000,
                                       prefix="iqtree_boot")
    iqtree_features_df = get_splits_df(msa_path= msa_path,true_tree_path= None, model=model,
                                                           bootstrap_tree_details=boot_tree_iqtree_details, program='iqtree', working_dir = tmp_files_folder
                                                           )

    logging.info("Running Fasttree")
    boot_tree_fasttree_details = fasttree_pipeline(tree_searches_folder, results_folder, msa_path, msa_type=msa_type,
                                           nb=0, model = model)
    fasttree_features_df = get_splits_df(msa_path=msa_path, true_tree_path=None, model=model,
                                       bootstrap_tree_details=boot_tree_fasttree_details, program='fasttree',
                                       working_dir=tmp_files_folder
                                       )

    logging.info("Running RAxML")
    boot_tree_raxml_details =  raxml_bootstrap_pipeline(tree_searches_folder ,results_folder , msa_path, prefix ="boot", model = model,  n_cpus=n_cpus,
                                                n_workers='auto')
    raxml_features_df = get_splits_df(msa_path=msa_path, true_tree_path=None, model=model,
                                         bootstrap_tree_details=boot_tree_raxml_details, program='raxml',
                                         working_dir=tmp_files_folder
                                         )

    return  iqtree_features_df, fasttree_features_df,raxml_features_df

def main():
    parser = job_parser()
    args = parser.parse_args()
    log_file_path = os.path.join(args.job_folder, 'log_file.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    study_folder = os.path.join(args.real_alignments_folder,args.job_study_name)
    if not os.path.exists(study_folder):
        tar_path = study_folder+'.tar.gz'
        logging.info(f"Extracting {tar_path} to {study_folder}")
        file = tarfile.open(tar_path)
        file.extractall(args.real_alignments_folder)
    msa_path_nexus = os.path.join(study_folder ,'alignment.nex')
    msa_path_fasta = os.path.join(study_folder ,'alignment.fasta')
    if not os.path.exists(msa_path_fasta):
        logging.info(f"Converting Nexus to Fasta")
        convert_nexus_to_fasta(msa_path_nexus, msa_path_fasta)
    job_working_dir = args.job_folder
    tree_searches_folder = os.path.join(job_working_dir, 'all_tree_searches')
    create_dir_if_not_exists(tree_searches_folder)
    results_folder = os.path.join(job_working_dir, 'results')
    create_dir_if_not_exists(results_folder)
    iqtree_features_df, fasttree_features_df, raxml_features_df = run_programs(msa_path_fasta, tree_searches_folder, results_folder, 'DNA', 'GTR+G', n_cpus= args.n_cpus)
    logging.info("Writing files to CSV")
    iqtree_features_df.to_csv(os.path.join(job_working_dir,"iqtree_real_data.csv"))
    fasttree_features_df.to_csv(os.path.join(job_working_dir, "fasttree_real_data.csv"))
    raxml_features_df.to_csv(os.path.join(job_working_dir, "raxml_real_data.csv"))
    logging.info("Done")

if __name__ == "__main__":
    main()
