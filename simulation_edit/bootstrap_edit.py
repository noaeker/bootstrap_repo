import sys
import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from side_code.basic_trees_manipulation import *
import dendropy
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from ete3 import *
import argparse
import random
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import bootstrap_MSA
from side_code.spr_prune_and_regraft import *
from side_code.file_handling import *
from side_code.raxml import generate_n_unique_tree_topologies_as_starting_trees, raxml_compute_tree_per_site_ll, generate_n_tree_topologies
from dendropy.calculate import treecompare
from side_code.code_submission import execute_command_and_write_to_log
from sklearn.metrics import silhouette_samples, silhouette_score






def get_trees_per_site_ll(trees, garbage_dir, msa_path, model):


    per_site_ll_scores = []
    for tree in trees:
        with open("tmp.tree", 'w') as TMP:
            TMP.write(tree)

        per_site_ll_score = raxml_compute_tree_per_site_ll(garbage_dir, full_data_path=msa_path,
                                                           tree_file="tmp.tree",
                                                           ll_on_data_prefix="per_site_ll",
                                                           model = model, opt = False)
        per_site_ll_scores.append(per_site_ll_score)
    final_mat = np.array(per_site_ll_scores)
    return final_mat
    # trees_total_ll = np.sum(final_mat, axis = 1)
    # correlations = []
    # for i in range(final_mat.shape[1]):
    #     col = final_mat[:,i]
    #     if np.var(col)==0:
    #         continue
    #     corr = pearsonr(col,trees_total_ll)[0]
    #     if not np.isnan(corr):
    #         correlations.append(corr)
    # agreement_min_max_ll = np.mean(best_vs_worst>0)
    # d = {f'{name}_agreement_min_max_ll': agreement_min_max_ll}
    # d.update(get_summary_statistics_dict(feature_name=f'{name}_site_corrs',values = correlations))
    # return d

def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)


def IQR(values):
    return np.percentile(values, 75) - np.percentile(values, 25)


def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean, 'median': np.mean, 'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max,
                                                             }):
    res = {}
    for func in funcs:
        res.update({f'{feature_name}_{func}': (funcs[func])(values)})
    return res


def get_booster_tree(taxa,mle_tree_path, comparison_tree, out_path ="booster.nw"):
    cmd = f"{BOOSTER_EXE} -a tbe -i {mle_tree_path} -b {comparison_tree} -@ 1 -o {out_path}"
    print(cmd)
    execute_command_and_write_to_log(cmd)
    with open(out_path) as B:
        bootster_tree = B.read()
    booster_dendro = get_tree_obj(bootster_tree, taxa)
    booster_tree_ete = Tree(newick=bootster_tree, format=0)
    add_internal_names(booster_tree_ete)
    return booster_dendro, booster_tree_ete

def get_pairwise_distances_mat(taxa, pdc):
    pairwise_distances = []
    for i, t1 in enumerate(taxa[:-1]):
        for t2 in taxa[i + 1:]:
            pairwise_distances.append(pdc(t1, t2))
    X = np.zeros((len(taxa), len(taxa)))
    triu = np.triu_indices(len(taxa), 1)
    X[triu] = pairwise_distances
    X = X.T
    X[triu] = X.T[triu]
    return X


def get_tree_obj(newick, taxa):
    tree = dendropy.Tree.get(data=newick, schema='newick', rooting='force-unrooted',
                             taxon_namespace=taxa)
    # print(tree.as_ascii_plot())
    tree.encode_bipartitions()
    # for edge in tree.postorder_edge_iter():
    #    edge.length = 1
    return tree


def get_list_of_taxa(node):
    res = []
    for leaf in node:
        res.append(leaf.name)
    return sorted(res)

def get_possible_moves(edges_list1,edges_list2):
    possible_moves = []
    for prune_edge in edges_list1:
        for rgft_edge in edges_list2:
                if not ((prune_edge.node_a == rgft_edge.node_a) or (prune_edge.node_b == rgft_edge.node_b) or (
                        prune_edge.node_b == rgft_edge.node_a) or (prune_edge.node_a == rgft_edge.node_b)):
                    possible_moves.append((prune_edge, rgft_edge))
    return possible_moves


# def get_spr_neighbors(tree,removed_node, remaining_tree):
#     removed_edges_list = []
#     for i, prune_node in enumerate(removed_node.iter_descendants("levelorder")):
#         if prune_node.up:
#             edge = Edge(node_a=prune_node.name, node_b=prune_node.up.name)
#             removed_edges_list.append(edge)
#     remaining_edges_list = []
#     for i, prune_node in enumerate(remaining_tree.iter_descendants("levelorder")):
#         if prune_node.up:
#             edge = Edge(node_a=prune_node.name, node_b=prune_node.up.name)
#             remaining_edges_list.append(edge)
#     possible_moves_mixed = get_possible_moves(remaining_edges_list,removed_edges_list)
#     random.shuffle(possible_moves_mixed)
#     tree_neighbors_mixed = [generate_neighbour(tree, possible_move) for possible_move in possible_moves_mixed[:20]]
#     return [tree.write(format=1) for tree in tree_neighbors_mixed]


def generate_booster_trees(taxa,mle_path,trees,garbage_dir,tree_tmp_path):
    all_tree_ete = []
    for tree in trees[:-1]:
        with open(tree_tmp_path, 'w') as TREE:
            TREE.write(tree)
        tree_dendro, tree_ete = get_booster_tree(taxa, mle_path, tree_tmp_path,
                                                      out_path=os.path.join(garbage_dir, "booster_pars.nw"))
        all_tree_ete.append(tree_ete)
    return all_tree_ete

def generate_bootstrap_booster_trees(msa_path, garbage_dir, n, model, taxa, mle_path):
    all_pars_ete_boot = []
    msa_out_path = os.path.join(garbage_dir,'boot_pars_msa.tmp')
    for i in range(n):
        b_msa = bootstrap_MSA(msa_path, msa_out_path)
        pars_path = generate_n_tree_topologies(1, b_msa,
                                   curr_run_directory=garbage_dir,
                                   seed=1, tree_type='pars', model=model)
        pars_dendro, pars_tree_ete = get_booster_tree(taxa, mle_path, pars_path,
                                                      out_path=os.path.join(garbage_dir, "booster_pars.nw"))
        all_pars_ete_boot.append(pars_tree_ete)
    return all_pars_ete_boot

def generate_partition_statistics(node,all_pars_ete,all_booster_ete,all_ML_ete, mle_tree_ete, best_ML_vs_true_tree_ete, pairwise_distances,
                                                               taxa,garbage_dir, msa_path, model):
        parsimony_support = [(pars_tree_ete & (node.name)).support for pars_tree_ete in all_pars_ete]
        parsimony_binary_support = [int((pars_tree_ete & (node.name)).support==1) for pars_tree_ete in all_pars_ete]
        boostrap_support = [(booster_tree_ete & (node.name)).support for booster_tree_ete in all_booster_ete]
        bootstrap_binary_support = [int((booster_tree_ete & (node.name)).support == 1) for booster_tree_ete in all_booster_ete]
        ML_tree_support = np.mean([(ML_tree_ete & (node.name)).support for ML_tree_ete in all_ML_ete])
        ML_tree_binary_support = [int((ML_tree_ete & (node.name)).support==1) for ML_tree_ete in all_ML_ete]
        true_support = (best_ML_vs_true_tree_ete&(node.name)).support
        true_binary_support = (best_ML_vs_true_tree_ete & (node.name)).support==1
        total_tree_divergence = get_tree_divergence(mle_tree_ete)
        mle_tree_ete_cp = mle_tree_ete.copy()
        node_cp = mle_tree_ete_cp & node.name
        removed_node = node_cp.detach()
        #labels = np.array([1 if t in get_list_of_taxa(removed_node) else 0 for t in
        #                                                      list([t.label for t in taxa])])
        #silhouete = metrics.silhouette_score(X=pairwise_distances, metric='precomputed',
        #                                     labels=np.array([1 if t in get_list_of_taxa(removed_node) else 0 for t in
        #                                                      list([t.label for t in taxa])]))

        mean_bl = np.mean(get_branch_lengths(mle_tree_ete))
        remaining_tree = mle_tree_ete_cp
        mle_tree_ete_orig = mle_tree_ete.copy()

        #lll_diffs.update(get_summary_statistics_dict(mixed_vs_mle,'mixed_vs_mle'))


        min_partition_divergence = min(get_tree_divergence(removed_node), get_tree_divergence(remaining_tree))
        divergence_ratio = min_partition_divergence/total_tree_divergence
        min_partition_size = min(len(get_list_of_taxa(removed_node)), len(get_list_of_taxa(remaining_tree)))
        partition_size_ratio = min_partition_size / len(taxa)

        statistics = {'partition_branch': node.dist, 'partition_branch_vs_mean': node.dist/mean_bl,'bootstrap_support': node.support, 'true_support': true_support, 'true_binary_support': true_binary_support, 'partition_size': min_partition_size,'partition_size_ratio': partition_size_ratio,
                    'partition_divergence': min_partition_divergence, 'divergence_ratio': divergence_ratio
                      }
        statistics.update(get_summary_statistics_dict(feature_name='pars_support', values  = parsimony_support))
        statistics.update(get_summary_statistics_dict(feature_name='bootstrap_support', values=boostrap_support))
        statistics.update(get_summary_statistics_dict(feature_name='ML_support', values= ML_tree_support))
        statistics.update(get_summary_statistics_dict(feature_name='pars_bi_support', values= parsimony_binary_support))
        statistics.update(get_summary_statistics_dict(feature_name='bootstrap_bi_support', values=bootstrap_binary_support))
        statistics.update(get_summary_statistics_dict(feature_name='ML_bi_support', values= ML_tree_binary_support))
        return statistics

def get_branch_lengths(tree):
    branch_lengths = []
    for node in tree.iter_descendants():
        # Do some analysis on node
        branch_lengths.append(node.dist)
    return branch_lengths

def get_tree_divergence(tree):
    branch_lengths = get_branch_lengths(tree)
    return np.sum(branch_lengths)


def get_file_rows(path):
    with open(path) as F:
        lines = F.readlines()
    return lines


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--msa_path',type = str, default= '/Users/noa/Workspace/simulations_results/raxml_grove_simulations/job_0/raxml_tree_0/52454/iqtree_msa_0/sim_msa.phy')
    parser.add_argument('--data_path', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/test/job_0/simulations_df.tsv")
    parser.add_argument('--final_output_path', type=str,
                        default="total_data.tsv")
    parser.add_argument('--work_path', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/bootstrap_edit_results')
    parser.add_argument('--n_pars', type=int, default=50)

    args = parser.parse_args()
    create_dir_if_not_exists(args.work_path)
    data = pd.read_csv(args.data_path, sep='\t')
    all_splits = pd.DataFrame()
    for true_tree_path in data['true_tree_path'].unique():
        taxa = dendropy.TaxonNamespace()
        tree_data = data.loc[data.true_tree_path == true_tree_path]
        for msa_path in tree_data['msa_path'].unique():
            bootstrap_tree_details = tree_data.loc[tree_data.msa_path == msa_path].head(1).squeeze()
            mle_path =  bootstrap_tree_details['final_tree_topology_path']
            all_mle_path = bootstrap_tree_details['all_final_tree_topologies_path']
            with open(mle_path, 'r') as NEWICK_PATH:
                mle_newick = NEWICK_PATH.read()
            #mle_tree_ete = generate_tree_object_from_newick(mle_newick, tree_type= 0)
            mle_tree_ete  = Tree(newick=mle_newick, format=0)
            add_internal_names(mle_tree_ete)
            mle_tree_dendro = get_tree_obj(mle_newick, taxa)
            b_pdc = mle_tree_dendro.phylogenetic_distance_matrix()
            pairwise_distances = get_pairwise_distances_mat(taxa, b_pdc)
            garbage_dir = os.path.join(args.work_path, 'garbage')
            create_dir_if_not_exists(garbage_dir)
            best_ML_vs_true_dendro, best_ML_vs_true_tree_ete = get_booster_tree(taxa, mle_path, true_tree_path,
                                                      out_path=os.path.join(garbage_dir,"booster_true.nw"))


            parsimony_trees_path = generate_n_tree_topologies(args.n_pars, bootstrap_tree_details["msa_path"],
                                                                                  curr_run_directory=garbage_dir,
                                                                                  seed=1, tree_type='pars', model= bootstrap_tree_details["model_short"])
            tree_tmp_path = os.path.join(garbage_dir, "tmp.tree")
            with open(parsimony_trees_path) as trees_path:
                parsimony_trees = trees_path.read().split("\n")[:-1]
            all_pars_ete=generate_booster_trees(taxa, mle_path, parsimony_trees, garbage_dir, tree_tmp_path)

            all_ML_nw = get_file_rows(all_mle_path)
            all_ML_ete = generate_booster_trees(taxa, mle_path, all_ML_nw, garbage_dir, tree_tmp_path)
            all_booster_ete = generate_bootstrap_booster_trees(msa_path,garbage_dir,args.n_pars,bootstrap_tree_details["model_short"],taxa,mle_path)
            for node in mle_tree_ete.iter_descendants():
                if not node.is_leaf():
                    statistics = generate_partition_statistics(node,all_pars_ete,all_booster_ete,all_ML_ete, mle_tree_ete, best_ML_vs_true_tree_ete, pairwise_distances,
                                                               taxa,garbage_dir, msa_path, bootstrap_tree_details["model_short"])
                    statistics.update(bootstrap_tree_details.to_dict())
                    all_splits = all_splits.append(statistics, ignore_index=True)
                    all_splits.to_csv(args.final_output_path, sep='\t')

            # sns.scatterplot(data=total_data, x='parsimony_support', y='Support',  s=30, alpha=0.6)
            # plt.show()


if __name__ == "__main__":
    main()