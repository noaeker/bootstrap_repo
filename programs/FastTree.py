from side_code.config import *
import os
import time
import shutil
from side_code.MSA_manipulation import bootstrap_MSA
from side_code.code_submission import execute_command_and_write_to_log
from ete3 import Tree

def get_booster_tree(mle_tree_path, comparison_trees, out_path):
    cmd = f"{BOOSTER_EXE} -a fbp -i {mle_tree_path} -b {comparison_trees} -@ 1 -o {out_path}"
    print(cmd)
    execute_command_and_write_to_log(cmd)


def fasttree_bootstrap_replicate(msa_path, msa_type, starting_tree, output_tree_file):
    if msa_type == 'DNA':
        model_str = '-nt -gtr'
    else:
        model_str = '-wag'

    search_command = f"{FASTTREE_EXE} -intree {starting_tree} {model_str} {msa_path}  > {output_tree_file}"
    execute_command_and_write_to_log(search_command)

    with open(output_tree_file,'r') as B:
        final_tree = B.read()
    return final_tree

def obtain_ML_fasttree(msa_path, msa_type, output_tree_file, model):
    if msa_type == 'DNA':
        if model=='JC':
            model_str = '-nt'
        else:
            model_str = '-nt -gtr'
    else:
        model_str = '-wag'

    search_command = f"{FASTTREE_EXE} {model_str} {msa_path} > {output_tree_file}"
    execute_command_and_write_to_log(search_command)
    return output_tree_file

def fasttree_pipeline(curr_run_dir, results_folder, msa_path, msa_type, nb, model):
    output_tree_file = os.path.join(results_folder,'fasttree_sh_bootstrap')
    obtain_ML_fasttree(msa_path, msa_type, output_tree_file, model)
    bootstrap_trees = []
    tmp_msa_path = os.path.join(curr_run_dir, 'tmp_msa.fasta')
    bootstrap_trees_file = os.path.join(curr_run_dir,'bootstrap_trees')
    st = time.time()
    for i in range(nb):
        bootstrap_msa = bootstrap_MSA(msa_path, tmp_msa_path)
        bootstrap_tree = fasttree_bootstrap_replicate(bootstrap_msa, msa_type, output_tree_file, output_tree_file = os.path.join(curr_run_dir,f'bootstrap.tree'))
        bootstrap_trees.append(bootstrap_tree)
    end = time.time()
    bootsrtap_time = end-st
    with open(bootstrap_trees_file,'w') as BF:
        BF.write("\n".join(bootstrap_trees))
    bootstrap_output_path = os.path.join(results_folder,'fasttree_standard_bootstrap')

    get_booster_tree(output_tree_file, bootstrap_trees_file, out_path =bootstrap_output_path)
    return {'sh_bootstrap': output_tree_file, 'standard_bootstrap': bootstrap_output_path,'boot_run_time': bootsrtap_time}

#t = fasttree_pipeline(curr_run_dir= os.path.join(os.getcwd(), 'trash'),msa_path="/Users/noa/Workspace/bootstrap_results/test/job_0/raxml_tree_0/25117/iqtree_msa_0/sim_msa.fa", msa_type='DNA', nb = 30)
#print(t)