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
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from ete3 import *
import argparse
from side_code.basic_trees_manipulation import *
from side_code.file_handling import *
from side_code.raxml import generate_n_unique_tree_topologies_as_starting_trees, raxml_compute_tree_per_site_ll, generate_n_tree_topologies
from dendropy.calculate import treecompare
from side_code.code_submission import execute_command_and_write_to_log
from sklearn.metrics import silhouette_samples, silhouette_score


# per_site_ll_scores = []
# for tree in parsimony_trees:
#     with open("tmp.tree", 'w') as TMP:
#         TMP.write(tree)
#
#     per_site_ll_score = raxml_compute_tree_per_site_ll(garbage_dir, full_data_path=msa_path,
#                                                        tree_file="tmp.tree",
#                                                        ll_on_data_prefix="per_site_ll",
#                                                        model=bootstrap_tree["model_short"], opt = False)
#     per_site_ll_scores.append(per_site_ll_score)
# final_mat = np.array(per_site_ll_scores)



def get_trees_per_site_ll_agreement(trees, garbage_dir, msa_path, model):


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
    trees_total_ll = np.sum(final_mat, axis = 1)
    best_tree_ind = np.argmax(trees_total_ll)
    worst_tree_ind = np.argmin(trees_total_ll)
    best_vs_worst = final_mat[best_tree_ind:,]-final_mat[worst_tree_ind:,]
    return max(np.mean(best_vs_worst>0), np.mean(best_vs_worst<0))

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
    execute_command_and_write_to_log(cmd)
    with open(out_path) as B:
        bootster_tree = B.read()
    booster_dendro = get_tree_obj(bootster_tree, taxa)
    booster_tree_ete = generate_tree_object_from_newick(bootster_tree, format=0)
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



def generate_partition_statistics(node, mle_tree_ete, best_ML_vs_true_tree_ete, pairwise_distances, all_pars_ete,
                                                           taxa, all_ML_ete):
        parsimony_support = [(pars_tree_ete & (node.name)).support for pars_tree_ete in all_pars_ete]
        parsimony_binary_support = [int((pars_tree_ete & (node.name)).support==1) for pars_tree_ete in all_pars_ete]
        ML_tree_support = np.mean([(ML_tree_ete & (node.name)).support for ML_tree_ete in all_ML_ete])
        ML_tree_binary_support = [int((ML_tree_ete & (node.name)).support==1) for ML_tree_ete in all_ML_ete]
        true_support = (best_ML_vs_true_tree_ete&(node.name)).support
        true_binary_support = (best_ML_vs_true_tree_ete & (node.name)).support==1
        total_tree_divergence = get_tree_divergence(mle_tree_ete)
        mle_tree_ete_cp = mle_tree_ete.copy()
        node_cp = mle_tree_ete_cp & node.name
        removed_node = node_cp.detach()
        silhouete = metrics.silhouette_score(X=pairwise_distances, metric='precomputed',
                                             labels=np.array([1 if t in get_list_of_taxa(removed_node) else 0 for t in
                                                              list([t.label for t in taxa])]))

        mean_bl = np.mean(get_branch_lengths(mle_tree_ete))
        remaining_tree = mle_tree_ete_cp
        min_partition_divergence = min(get_tree_divergence(removed_node), get_tree_divergence(remaining_tree))
        divergence_ratio = min_partition_divergence/total_tree_divergence
        min_partition_size = min(len(get_list_of_taxa(removed_node)), len(get_list_of_taxa(remaining_tree)))
        partition_size_ratio = min_partition_size / len(taxa)

        statistics = {'partition_branch': node.dist, 'partition_branch_vs_mean': node.dist/mean_bl,'bootstrap_support': node.support, 'true_support': true_support, 'true_binary_support': true_binary_support, 'Silhouette': silhouete, 'partition_size': min_partition_size,'partition_size_ratio': partition_size_ratio,
                    'partition_divergence': min_partition_divergence, 'divergence_ratio': divergence_ratio
                      }
        statistics.update(get_summary_statistics_dict(feature_name='pars_support', values  = parsimony_support))
        statistics.update(get_summary_statistics_dict(feature_name='ML_support', values= ML_tree_support))
        statistics.update(get_summary_statistics_dict(feature_name='pars_bi_support', values= parsimony_binary_support))
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
    parser.add_argument('--n_workers', type=int, default=4)

    args = parser.parse_args()

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
            mle_tree_ete = generate_tree_object_from_newick(mle_newick)
            mle_tree_dendro = get_tree_obj(mle_newick, taxa)

            b_pdc = mle_tree_dendro.phylogenetic_distance_matrix()
            pairwise_distances = get_pairwise_distances_mat(taxa, b_pdc)
            garbage_dir = os.path.join(os.getcwd(), 'garbage')
            create_dir_if_not_exists(garbage_dir)


            best_ML_vs_true_dendro, best_ML_vs_true_tree_ete = get_booster_tree(taxa, mle_path, true_tree_path,
                                                      out_path=os.path.join(garbage_dir,"booster_true.nw"))


            parsimony_trees_path = generate_n_tree_topologies(50, bootstrap_tree_details["msa_path"],
                                                                                  curr_run_directory=garbage_dir,
                                                                                  seed=1, tree_type='pars', model= bootstrap_tree_details["model_short"])
            with open(parsimony_trees_path) as trees_path:
                parsimony_trees = trees_path.read().split("\n")[:-1]
            all_pars_ete = []
            parsimony_tree_path = os.path.join(garbage_dir,"pars.tree")
            for parsimony_tree in parsimony_trees:
                with open(parsimony_tree_path, 'w') as PARS:
                    PARS.write(parsimony_tree)
                pars_dendro, pars_tree_ete = get_booster_tree(taxa, mle_path,parsimony_tree_path, out_path=os.path.join(garbage_dir,"booster_pars.nw"))
                all_pars_ete.append(pars_tree_ete)


            all_ML_nw = get_file_rows(all_mle_path)
            all_ML_ete = []
            tmp_ml_tree_file = os.path.join(garbage_dir,"tmp.ml.tree")
            for ML_tree in all_ML_nw:
                with open(tmp_ml_tree_file, 'w') as ML:
                    ML.write(ML_tree)
                ML_dendro, ML_tree_ete = get_booster_tree(taxa,  mle_path,tmp_ml_tree_file,
                                                          out_path=os.path.join(garbage_dir,"booster_ml.nw"))
                all_ML_ete.append(ML_tree_ete)
            for node in mle_tree_ete.iter_descendants():
                if not node.is_leaf():
                    statistics = generate_partition_statistics(node, mle_tree_ete, best_ML_vs_true_tree_ete, pairwise_distances, all_pars_ete,
                                                               taxa, all_ML_ete)
                    statistics.update(bootstrap_tree_details.to_dict())
                    all_splits = all_splits.append(statistics, ignore_index=True)
                    all_splits.to_csv(args.final_output_path, sep='\t')

            # sns.scatterplot(data=total_data, x='parsimony_support', y='Support',  s=30, alpha=0.6)
            # plt.show()


if __name__ == "__main__":
    main()