import sys
import re
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
import numpy as np
import tarfile
import logging
from Bio import SeqIO
import shutil
import pickle



def run_programs(msa_path,tree_searches_folder, results_folder,msa_type, model, n_cpus,use_existing_trees, program):
    tmp_files_folder = os.path.join(tree_searches_folder, 'tmp')
    create_dir_if_not_exists(tmp_files_folder)
    # logging.info("Running IQTREE")
    # boot_tree_iqtree_details = iqtree_pipeline(tree_searches_folder, results_folder, msa_path, model=model, nb=1000,
    #                                    prefix="iqtree_boot", n_cpus = n_cpus)
    # iqtree_features_df = get_splits_df(msa_path= msa_path,true_tree_path= None, model=model,
    #                                                        bootstrap_tree_details=boot_tree_iqtree_details, program='iqtree', working_dir = tmp_files_folder,n_cpus = n_cpus
    #                                                        )
    #
    # logging.info("Running Fasttree")
    # boot_tree_fasttree_details = fasttree_pipeline(tree_searches_folder, results_folder, msa_path, msa_type=msa_type,
    #                                        nb=0, model = model)
    # fasttree_features_df = get_splits_df(msa_path=msa_path, true_tree_path=None, model=model,
    #                                    bootstrap_tree_details=boot_tree_fasttree_details, program='fasttree',
    #                                    working_dir=tmp_files_folder,n_cpus = n_cpus
    #                                    )


    if program=='raxml':
        boot_tree_raxml_details =  raxml_bootstrap_pipeline(tree_searches_folder ,results_folder , msa_path, prefix ="boot", model = model,  n_cpus=n_cpus,
                                                    n_workers='auto', use_existing_trees = use_existing_trees)
        raxml_features_df,raxml_obj_with_features = get_splits_df(msa_path=msa_path, true_tree_path=None, model=model,
                                             bootstrap_tree_details=boot_tree_raxml_details, program='raxml',
                                             working_dir=tmp_files_folder, n_cpus = min(n_cpus, 20))
        return raxml_features_df, raxml_obj_with_features
    elif program=='iqtree':
     boot_tree_iqtree_details = iqtree_pipeline(tree_searches_folder, results_folder, msa_path, model=model, nb=1000,
                                        prefix="iqtree_boot", n_cpus = n_cpus)
     iqtree_features_df,iqtree_obj_with_features = get_splits_df(msa_path= msa_path,true_tree_path= None, model=model,
                                                            bootstrap_tree_details=boot_tree_iqtree_details, program='iqtree', working_dir = tmp_files_folder,n_cpus = n_cpus
                                                            )
     return iqtree_features_df,iqtree_obj_with_features



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
        records = SeqIO.parse(msa_path_nexus, "nexus")
        valid_records = []
        for i,record in enumerate(records):
            len_seq = len(str(record.seq).replace('-','').replace('?',''))
            if (len_seq)>0:
                if args.change_names:
                    record.name=f'taxa_{i}'
                    record.id=f'taxa_{i}'
                else:
                    record.name = record.name.replace('@','_')
                    record.id = record.id.replace('@', '_')
                valid_records.append(record)
        count = SeqIO.write(valid_records, msa_path_fasta, "fasta")
        print("Converted %i records" % count)
        #convert_nexus_to_fasta(msa_path_nexus, msa_path_fasta)
    job_working_dir = args.job_folder
    tree_searches_folder = os.path.join(job_working_dir, 'all_tree_searches')
    create_dir_if_not_exists(tree_searches_folder)
    results_folder = os.path.join(job_working_dir, 'results')
    create_dir_if_not_exists(results_folder)
    raxml_features_df,raxml_obj_with_features = run_programs(msa_path_fasta, tree_searches_folder, results_folder, 'Protein', args.model, n_cpus= args.n_cpus,use_existing_trees = args.use_existing_trees, program = args.program)
    logging.info("Writing files to CSV")
    #iqtree_features_df.to_csv(os.path.join(job_working_dir,"iqtree_real_data.csv"))
    #fasttree_features_df.to_csv(os.path.join(job_working_dir, "fasttree_real_data.csv"))
    raxml_features_df.to_csv(os.path.join(job_working_dir, "raxml_real_data.csv"))
    raxml_obj_with_features.write(format=1, outfile=os.path.join(job_working_dir,"new_tree.nw"))
    logging.info("Done")

if __name__ == "__main__":
    main()
