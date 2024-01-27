import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from simulation_edit.simulations_edit_argparser import job_parser
import dendropy
import numpy as np
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import bootstrap_MSA, get_MSA_seq_names, add_unique_seq
from side_code.spr_prune_and_regraft import *
from simulation_edit.msa_features import get_msa_stats
from simulation_edit.trees_features_extraction_helper import get_tree_divergence, get_branch_lengths, mad_tree_parameter,get_neighboring_nodes,get_bootstrap_support, get_nni_neighbors
from side_code.file_handling import *
from programs.raxml import raxml_compute_tree_per_site_ll, generate_n_tree_topologies, remove_redundant_sequences, \
    raxml_optimize_trees_for_given_msa
from side_code.code_submission import execute_command_and_write_to_log
from scipy.stats import kurtosis
from scipy.stats import skew





def get_mle_tree_features(working_dir, msa_path, model, mle_tree_path, garbage_dir, raxml_path, mad_path):
    mle_tree_obj = Tree(mle_tree_path, format=1)
    branch_lengths = get_branch_lengths(mle_tree_obj)
    curr_pruned_tree_path, curr_pruned_msa_path = get_pruned_tree_and_msa(working_dir, msa_path, mle_tree_obj)
    orig_tree_ll = raxml_optimize_trees_for_given_msa(curr_pruned_msa_path, ll_on_data_prefix="orig_tree_ll",
                                                      tree_file=curr_pruned_tree_path,
                                                      curr_run_directory=garbage_dir,
                                                      model=model,
                                                      opt_model_and_brlen=True, n_cpus=1, n_workers='auto',
                                                      return_opt_tree=False,
                                                      program_path = raxml_path
                                                      )
    st = time.time()
    stats = {
        'orig_tree_ll': orig_tree_ll,
        'feature_tree_mad_score': mad_tree_parameter(mle_tree_path, program_path=mad_path),
        'feature_total_tree_divergence': get_tree_divergence(mle_tree_obj),
             'feature_25_pct_bl': np.percentile(branch_lengths, 25),
             'feature_75_pct_bl': np.percentile(branch_lengths, 75),
             'feature_median_bl': np.median(branch_lengths), 'feature_var_bl': np.var(branch_lengths),
             'feature_skew_bl': skew(branch_lengths), 'feature_kurtosis_bl': kurtosis(branch_lengths)}
    end = time.time()
    return stats, end-st




def get_pruned_tree_and_msa(curr_run_dir, msa_path, mle_tree_ete):
    pruned_msa_path = os.path.join(curr_run_dir, "pruned_msa.fasta")
    pruned_tree_path = os.path.join(curr_run_dir, "pruned_tree.nw")
    add_unique_seq(msa_path, pruned_msa_path)
    pruned_tree = mle_tree_ete.copy()
    pruned_tree.prune(get_MSA_seq_names(pruned_msa_path))
    pruned_tree.write(outfile=pruned_tree_path, format=1)
    return pruned_tree_path, pruned_msa_path


def get_nni_statistics(working_dir, orig_tree_ll, nni_neighbors, msa_path, model, garbage_dir, program_path = None):
    all_neig_ll = []
    neighbors_tmp_path = os.path.join(garbage_dir,'tmp_neigh.tree')
    for neighbor in nni_neighbors:
        neighbor.write(format=1, outfile=neighbors_tmp_path)
        neighbor = Tree(neighbors_tmp_path, format=1)
        curr_pruned_tree_path, curr_pruned_msa_path = get_pruned_tree_and_msa(working_dir, msa_path,
                                                                               neighbor)
        neighbor_ll, optimized_tree = raxml_optimize_trees_for_given_msa(curr_pruned_msa_path,
                                                                         ll_on_data_prefix="nni_neighbors",
                                                                         tree_file=curr_pruned_tree_path,
                                                                         curr_run_directory=garbage_dir, model=model,
                                                                         opt_model_and_brlen=True, n_cpus=1,
                                                                         n_workers='auto', return_opt_tree=True, program_path= program_path
                                                                         )

        all_neig_ll.append(neighbor_ll)

    end_ll = time.time()
    min_ll_diff = orig_tree_ll - np.max(all_neig_ll)
    max_ll_diff = orig_tree_ll - np.min(all_neig_ll)

    abayes_metric = (orig_tree_ll) / (((orig_tree_ll) + (all_neig_ll[0]) + (all_neig_ll[1])) / 3)
    nni_statistics = {
                      f'feature_abayes': abayes_metric, f'feature_min_ll_diff': min_ll_diff,
                      f'feature_max_ll_diff': max_ll_diff,
                      }

    return nni_statistics



def get_partition_statistics(node, mle_tree_obj, extra_support_features_dict, fbp_true_support_tree, bootstrap_support_trees_dict):
    mle_tree_obj_cp = mle_tree_obj.copy()
    tree_divergence = get_tree_divergence(mle_tree_obj)
    node_cp = mle_tree_obj_cp & node.name
    removed_node = node_cp.detach()
    remaining_tree = mle_tree_obj_cp
    min_partition_divergence = min(get_tree_divergence(removed_node), get_tree_divergence(remaining_tree))
    divergence_ratio = min_partition_divergence / tree_divergence
    min_partition_size = min(len((removed_node.get_tree_root())), len((remaining_tree.get_tree_root())))
    partition_size_ratio = min_partition_size / len((mle_tree_obj.get_tree_root()))
    neighboring_node_pairs = get_neighboring_nodes(mle_tree_obj, node.name)
    bipart_branch_lengths = [node.dist for node in neighboring_node_pairs[0]] + [node.dist for node in
                                                                             neighboring_node_pairs[1]]

    stats = {'bootstrap_support': node.support,
             'feature_partition_size': min_partition_size,
             'feature_partition_divergence':  min_partition_divergence ,
        'feature_partition_size_ratio': partition_size_ratio, 'feature_divergence_ratio': divergence_ratio,
             'feature_mean_bl_bipart': np.mean(bipart_branch_lengths),
             'feature_min_bl_bipart': np.min(bipart_branch_lengths),
             'feature_max_bl_bipart': np.max(bipart_branch_lengths),
             'feature_min_vs_max_bl_bipart': np.min(bipart_branch_lengths)/np.max(bipart_branch_lengths),
             'feature_var_bl_bipart': np.var(bipart_branch_lengths) / np.max(bipart_branch_lengths),
             'feature_partition_branch_vs_mean': node.dist / tree_divergence,
             'feature_partition_branch': node.dist,
             'feature_partition_branch_vs_bipart_mean': node.dist / np.mean(bipart_branch_lengths)}

    for extra_support in extra_support_features_dict:
        stats[f"feature_{extra_support}"] = ((extra_support_features_dict[extra_support])&node.name).support
        node_neighbors_supports = [((extra_support_features_dict[extra_support])&node_n.name).support for node_n in neighboring_node_pairs[0]+neighboring_node_pairs[1] if not node_n.is_leaf()]
        stats.update({f'feature_{extra_support}_mean_bipart': np.mean(node_neighbors_supports),f'feature_{extra_support}_min_bipart': np.min(node_neighbors_supports)})

    if fbp_true_support_tree is not None:
        stats["true_support"] = ((fbp_true_support_tree) & node.name).support #update true support
    for bootstrap_support in bootstrap_support_trees_dict:
        stats[f"{bootstrap_support}"] = ((bootstrap_support_trees_dict[bootstrap_support])& node.name).support
    return stats

def extract_all_features_per_mle(working_dir, msa_path, model, mle_tree_path, extra_bootstrap_support_paths, all_mles_tree_path = None, true_tree_path = None, booster_program_path = None, raxml_program_path = None, mad_program_path = None):
    mle_tree_obj = Tree(mle_tree_path, format=0)
    add_internal_names(mle_tree_obj)
    mle_with_internal_path = os.path.join(working_dir, "mle_with_internal.nw")
    mle_tree_obj.write(outfile=mle_with_internal_path, format=1)
    garbage_dir = os.path.join(working_dir, 'tmp')
    create_dir_if_not_exists(garbage_dir)
    mle_tree_feautres,mle_tree_features_time = get_mle_tree_features(working_dir, msa_path, model, mle_with_internal_path, garbage_dir, mad_path= mad_program_path, raxml_path= raxml_program_path)
    st_extra_features = time.time()
    msa_features = get_msa_stats(msa_path,model)
    parsimony_trees_path = generate_n_tree_topologies(n = 100, msa_path = msa_path,
                                                      curr_run_directory=garbage_dir,
                                                      seed=1, tree_type='pars',
                                                      model=model)
    tbe_pars_support_tree,fbp_pars_support_tree = get_bootstrap_support(garbage_dir, mle_with_internal_path, parsimony_trees_path, program_path = booster_program_path)
    extra_support_features_dict = {'tbe_pars': tbe_pars_support_tree, 'fbp_pars': fbp_pars_support_tree}
    if all_mles_tree_path is not None:
        tbe_MLEs_support_tree, fbp_MLEs_support_tree = get_bootstrap_support(garbage_dir, mle_with_internal_path, all_mles_tree_path, program_path= booster_program_path)
        extra_support_features_dict.update({'tbe_MLEs':tbe_MLEs_support_tree,'fbp_MLEs':fbp_MLEs_support_tree})
    end_extra_features = time.time()
    extra_features_time = end_extra_features-st_extra_features
    per_node_nni_time = 0
    per_node_other_time = 0
    if true_tree_path is not None:
        tbe_true_support_tree, fbp_true_support_tree = get_bootstrap_support(garbage_dir, mle_with_internal_path,
                                                                             true_tree_path)


    else:
        fbp_true_support_tree = None
    bootstrap_support_trees_dict = {}
    for bootstrap_support in extra_bootstrap_support_paths:
        bootstrap_tree_obj = Tree(newick=extra_bootstrap_support_paths[bootstrap_support], format=0)
        add_internal_names(bootstrap_tree_obj)
        bootstrap_support_trees_dict[bootstrap_support] = bootstrap_tree_obj
    all_splits = pd.DataFrame()
    for node in mle_tree_obj.iter_descendants():
        if not node.is_leaf():
            st_nni = time.time()
            nni_neighobrs = get_nni_neighbors(mle_with_internal_path,node.name )
            NNI_stats = get_nni_statistics(working_dir, mle_tree_feautres["orig_tree_ll"], nni_neighobrs , msa_path,
                                           model, garbage_dir, program_path = raxml_program_path)
            end_nni = time.time()
            per_node_nni_time+=end_nni-st_nni
            partition_statistics = get_partition_statistics(node, mle_tree_obj, extra_support_features_dict,fbp_true_support_tree,bootstrap_support_trees_dict)

            end_partition_statistics = time.time()
            per_node_other_time+= end_partition_statistics-end_nni
            partition_statistics.update({'node_name': node.name})
            partition_statistics.update(msa_features)
            partition_statistics.update(mle_tree_feautres)
            partition_statistics.update(NNI_stats)
            node.add_feature("partition_statistics" , partition_statistics)
            all_splits = all_splits.append(partition_statistics, ignore_index= True)
    all_splits["mle_tree_FEATURE_time"] = mle_tree_features_time
    all_splits["extra_FEATURE_extraction_time"] = extra_features_time
    all_splits["total_nni_per_node_FEATURE_extraction_time"] = per_node_nni_time
    all_splits["total_other_per_node_FEATURE_extraction_time"] = per_node_other_time
    return mle_tree_obj,all_splits




