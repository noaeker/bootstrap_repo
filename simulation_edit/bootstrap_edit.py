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
from side_code.MSA_manipulation import bootstrap_MSA,get_MSA_seq_names, add_unique_seq
from side_code.spr_prune_and_regraft import *
from side_code.file_handling import *
from programs.raxml import raxml_compute_tree_per_site_ll, generate_n_tree_topologies,remove_redundant_sequences,raxml_optimize_trees_for_given_msa
from side_code.code_submission import execute_command_and_write_to_log
from math import exp





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


def get_booster_tree(mle_tree_path, comparison_tree, out_path ="booster.nw"):
    cmd = f"{BOOSTER_EXE} -a tbe -i {mle_tree_path} -b {comparison_tree} -@ 1 -o {out_path}"
    execute_command_and_write_to_log(cmd)
    with open(out_path) as B:
        bootster_tree = B.read()
    booster_tree_ete = Tree(newick=bootster_tree, format=0)
    add_internal_names(booster_tree_ete)
    return booster_tree_ete

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




def generate_booster_trees(mle_path,trees,garbage_dir,tree_tmp_path):
    all_tree_ete = []
    for tree in trees[:-1]:
        with open(tree_tmp_path, 'w') as TREE:
            TREE.write(tree)
        tree_ete = get_booster_tree(mle_path, tree_tmp_path,
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



def get_neighboring_nodes(tree_ete, node_name):
    tree_ete_cp = tree_ete.copy()
    tree_ete_cp.set_outgroup(tree_ete_cp & node_name)
    children_nodes = [node_c for node_c in (tree_ete_cp & node_name).children]
    sister_children_nodes = \
        ([node_c for node_c in (tree_ete_cp & node_name).up.children if node_c.name != node_name][0]).children
    return children_nodes+sister_children_nodes

def get_node_support_feautres_among_tree_groups(node_name, trees, name):
    tree_support = ([(tree_ete & (node_name)).support for tree_ete in trees ])
    tree_binary_support = [int((tree_ete & (node_name)).support == 1) for tree_ete in trees]
    neighbors_ete_support_values = []
    neighbors_ete_support_values_binary = []
    for tree_ete in trees:
        neighboring_nodes = get_neighboring_nodes(tree_ete, node_name)
        support_values = [node.support for node in neighboring_nodes]
        binary_support_values = [int(node.support == 1) for node in neighboring_nodes]
        neighbors_ete_support_values+= support_values
        neighbors_ete_support_values_binary+= binary_support_values

    features = {f"feature_mean_{name}": np.mean(tree_support),
                f"feature_mean_{name}_binary": np.mean(tree_binary_support),f"feature_mean_{name}_neighbors": np.mean(neighbors_ete_support_values), f"feature_mean_{name}_neighbors_binary": np.mean(neighbors_ete_support_values_binary) }


    return features





def generate_partition_statistics(node, mle_tree_obj, extra_tree_groups, extra_boot_ete, best_ML_vs_true_tree_ete,
                                  ):
        statistics = {}

        for tree_group in extra_tree_groups:

            statistics.update(get_node_support_feautres_among_tree_groups(node.name, extra_tree_groups[tree_group], name = tree_group))


        true_support = (best_ML_vs_true_tree_ete&(node.name)).support
        true_binary_support = (best_ML_vs_true_tree_ete & (node.name)).support==1
        total_tree_divergence = get_tree_divergence(mle_tree_obj)
        mle_tree_ete_cp = mle_tree_obj.copy()
        node_cp = mle_tree_ete_cp & node.name
        removed_node = node_cp.detach()
        mean_bl = np.mean(get_branch_lengths(mle_tree_obj))
        var_bl = np.var(get_branch_lengths(mle_tree_obj))
        remaining_tree = mle_tree_ete_cp
        min_partition_divergence = min(get_tree_divergence(removed_node), get_tree_divergence(remaining_tree))
        divergence_ratio = min_partition_divergence/total_tree_divergence
        min_partition_size = min(len((removed_node.get_tree_root())), len((remaining_tree.get_tree_root())))
        partition_size_ratio = min_partition_size / len((mle_tree_obj.get_tree_root()))
        neighboring_nodes= get_neighboring_nodes(mle_tree_obj, node.name)
        childs_brlen = [c_node.dist for c_node in neighboring_nodes]

        statistics.update({'feature_mean_bl': mean_bl,'feature_var_bl': var_bl,'feature_mean_neighbor_brlen': np.mean(childs_brlen),'feature_min_neighbor_brlen': np.min(childs_brlen),'feature_partition_branch': node.dist,'feature_partition_branch_vs_mean': node.dist/mean_bl,'bootstrap_support': node.support, 'true_support': true_support, 'true_binary_support': true_binary_support, 'feature_partition_size': min_partition_size,'feature_partition_size_ratio': partition_size_ratio,
                    'feature_partition_divergence': min_partition_divergence, 'feature_divergence_ratio': divergence_ratio})
        for b_method in extra_boot_ete:
            statistics.update({f'feature_{b_method}_support':(extra_boot_ete[b_method]&node.name).support})
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



def get_program_default_ML_tree(program):
    if program =='raxml':
        name = 'final_tree_topology_path'
    elif program =='iqtree':
        name = 'final_tree_ultrafast'
    elif program=='fasttree':
        name = 'sh_bootstrap'
    return name


def get_bootstrap_and_tree_groups(program, bootstrap_tree_details,mle_path,garbage_dir,n_pars):
    parsimony_trees_path = generate_n_tree_topologies(n_pars, bootstrap_tree_details["msa_path"],
                                                      curr_run_directory=garbage_dir,
                                                      seed=1, tree_type='pars',
                                                      model=bootstrap_tree_details["tree_search_model"])
    tree_tmp_path = os.path.join(garbage_dir, "tmp.tree")
    with open(parsimony_trees_path) as trees_path:
        parsimony_trees = trees_path.read().split("\n")[:-1]
    all_pars_ete = generate_booster_trees(mle_path, parsimony_trees, garbage_dir, tree_tmp_path)
    extra_tree_groups = {'parsimony_trees': all_pars_ete}
    extra_boot = {}
    if program == 'raxml':
        all_mle_path = bootstrap_tree_details['all_final_tree_topologies_path']
        all_ML_nw = get_file_rows(all_mle_path)
        all_ML_ete = generate_booster_trees(mle_path, all_ML_nw, garbage_dir, tree_tmp_path)
        extra_tree_groups.update( {'all_ML_boot_raxml': all_ML_ete})
    elif program == 'iqtree':
        final_tree_aLRT = bootstrap_tree_details['final_tree_aLRT']
        aLRT_ete = Tree(newick=final_tree_aLRT, format=0)
        add_internal_names(aLRT_ete)
        final_tree_aBayes_path = bootstrap_tree_details['final_tree_aBayes']
        aBayes_ete = Tree(newick=final_tree_aBayes_path, format=0)
        add_internal_names(aBayes_ete)
        extra_boot = {'aLRT_iqtree': aLRT_ete, 'aBayes_iqtree': aBayes_ete}
    elif program == 'fasttree':
        standard_bootstrap = bootstrap_tree_details['standard_bootstrap']
        if os.path.exists(standard_bootstrap):
            standard_ete = Tree(newick=standard_bootstrap, format=0)
            add_internal_names(standard_ete)
            extra_boot = {'standard_fasttree_boot': standard_ete}

    return extra_boot, extra_tree_groups



def generate_tree(node_pair1, node_pair2):
    sub_tree1 = Tree()
    sub_tree1.add_child(node_pair1[0])
    sub_tree1.add_child(node_pair1[1])
    sub_tree2 = Tree()
    sub_tree2.add_child(node_pair2[0])
    sub_tree2.add_child(node_pair2[1])
    final_tree = Tree()
    final_tree.add_child(sub_tree1)
    final_tree.add_child(sub_tree2)
    return final_tree

def get_nni_neighbors(tree_path,node_name):
    tree = Tree(tree_path, format = 1)
    tree.set_outgroup(tree&node_name)
    node_sister = [node for node in (tree&node_name).up.children if node.name!=node_name][0]
    cousins = node_sister.children
    kids = (tree & node_name).children
    pruned_first_child = (kids[0]).detach()
    pruned_second_child = (kids[0]).detach()
    pruned_first_cousin = (cousins[0]).detach()
    pruned_second_cousin = (cousins[0]).detach()
    first_final_tree = generate_tree((pruned_first_child,pruned_first_cousin),(pruned_second_child,pruned_second_cousin))
    second_final_tree = generate_tree((pruned_first_child,pruned_second_cousin),(pruned_second_child,pruned_first_cousin))
    neighbors = (first_final_tree, second_final_tree)
    return neighbors



def check_tree_is_ok(tree):
    if len(tree.get_tree_root().children)>3:
        return False
    for node in tree.iter_descendants():
        if len(node.children)>2:
            return False
    return True



def get_pruned_tree_and_msa(curr_run_dir, msa_path, model,mle_tree_ete):
    pruned_msa_path = os.path.join(curr_run_dir, "pruned_msa.fasta")
    pruned_tree_path = os.path.join(curr_run_dir, "pruned_tree.nw")
    add_unique_seq(msa_path, pruned_msa_path)
    pruned_tree = mle_tree_ete.copy()
    pruned_tree.prune(get_MSA_seq_names(pruned_msa_path))
    pruned_tree.write(outfile=pruned_tree_path, format=1)
    return pruned_tree_path, pruned_msa_path

#model=bootstrap_tree_details_dict["model_short"]




def get_nni_statistics(orig_tree_ll, orig_tree_mean_bl,nni_neighbors,neighbors_tmp_path,curr_run_dir, msa_path,bootstrap_tree_details_dict,garbage_dir, opt  = False, simple_model = False):
    st_ll = time.time()
    all_neig_ll = []
    mean_branch_lengths = []
    for neighbor in nni_neighbors:
        neighbor.write(format=1, outfile=neighbors_tmp_path)
        neighbor = Tree(neighbors_tmp_path, format=1)
        if simple_model:
            model = 'JC'
        else:
            model = bootstrap_tree_details_dict["tree_search_model"]
        curr_pruned_tree_path, curr_pruned_msa_path = get_pruned_tree_and_msa(curr_run_dir, msa_path,
                                                                              bootstrap_tree_details_dict[
                                                                                  "tree_search_model"], neighbor)
        neighbor_ll, optimized_tree = raxml_optimize_trees_for_given_msa(curr_pruned_msa_path,  ll_on_data_prefix="nni_neighbors", tree_file=curr_pruned_tree_path,
                                       curr_run_directory = garbage_dir, model = model, opt_model_and_brlen=opt,n_cpus = 1, n_workers = 'auto', return_opt_tree = True
                                       )

        all_neig_ll.append(neighbor_ll)
        if opt:
            mle_tree_ete = Tree(optimized_tree, format=1)
            mean_branch_length = np.mean(get_branch_lengths( mle_tree_ete))
            mean_branch_lengths.append(mean_branch_length )

    end_ll = time.time()
    neig_ll_evaluation_time = end_ll - st_ll
    min_ll_diff = orig_tree_ll - np.max(all_neig_ll)
    max_ll_diff = orig_tree_ll - np.min(all_neig_ll)

    abayes_metric = (orig_tree_ll) / (((orig_tree_ll) + (all_neig_ll[0]) + (all_neig_ll[1])) / 3)
    name = f"opt={opt}_model_{simple_model}"
    nni_statistics = {f'total_time_{name}': neig_ll_evaluation_time,
        f'feature_abayes_{name}': abayes_metric, f'orig_tree_ll_{name}': orig_tree_ll, f'feature_min_ll_diff_{name}': min_ll_diff, f'feature_max_ll_diff_{name}': max_ll_diff,
}
    if opt:
        nni_statistics.update({f'feature_min_mean_branch_length_{name}': orig_tree_mean_bl-np.min(mean_branch_lengths), f'feature_max_mean_branch_length_{name}': orig_tree_mean_bl-np.max(mean_branch_lengths)})

    return nni_statistics, neig_ll_evaluation_time


def msa_path_edit_analysis(msa_path, curr_run_dir, mle_path, true_tree_path, program, bootstrap_tree_details_dict, n_pars):

    msa_splits = pd.DataFrame()
    mle_tree_ete = Tree(mle_path, format=0)
    add_internal_names(mle_tree_ete)
    mle_with_internal_path = os.path.join(curr_run_dir, "mle_with_internal.nw")
    mle_tree_ete.write(outfile=mle_with_internal_path, format=1)
    mean_branch_length_orig = np.mean(get_branch_lengths(mle_tree_ete))

    garbage_dir = os.path.join(curr_run_dir, 'garbage')
    create_dir_if_not_exists(garbage_dir)
    st = time.time()
    best_ML_vs_true_tree_ete = get_booster_tree(mle_with_internal_path, true_tree_path,
                                                out_path=os.path.join(garbage_dir, "booster_true.nw"))

    extra_boot, extra_tree_groups = get_bootstrap_and_tree_groups(program, bootstrap_tree_details_dict, mle_with_internal_path,
                                                                  garbage_dir, n_pars)

    end = time.time()
    feature_extraction_time = end - st


    neighbors_tmp_path = os.path.join(curr_run_dir, "neighbors_tmp.nw")

    curr_pruned_tree_path, curr_pruned_msa_path = get_pruned_tree_and_msa(curr_run_dir, msa_path, bootstrap_tree_details_dict["tree_search_model"],mle_tree_ete)


    orig_tree_ll = raxml_optimize_trees_for_given_msa(curr_pruned_msa_path,  ll_on_data_prefix="orig_tree_ll", tree_file=curr_pruned_tree_path,
                                       curr_run_directory = garbage_dir, model = bootstrap_tree_details_dict["tree_search_model"], opt_model_and_brlen=True,n_cpus = 1, n_workers = 'auto', return_opt_tree = False
                                       )
    #orig_tree_ll_JC = raxml_optimize_trees_for_given_msa(curr_pruned_msa_path, ll_on_data_prefix="orig_tree_ll",
    #                                                  tree_file=curr_pruned_tree_path,
    #                                                  curr_run_directory=garbage_dir,
    #                                                  model="JC",
    #                                                  opt_model_and_brlen=True, n_cpus=1, n_workers='auto',
    #                                                  return_opt_tree=False
    #                                                  )

    total_neig_ll_evaluation_time_no_opt = 0
    total_neig_ll_evaluation_time_opt = 0
    for node in mle_tree_ete.iter_descendants():

        if not node.is_leaf():
            statistics = generate_partition_statistics(node, mle_tree_ete, extra_tree_groups, extra_boot,
                                                       best_ML_vs_true_tree_ete
                                                       )
            statistics["feature_n_unique_seq"] = len(get_MSA_seq_names(curr_pruned_msa_path))
            nni_neighbors = get_nni_neighbors(mle_with_internal_path, node.name)
            nni_statistics_no_opt,neig_ll_evaluation_time_no_opt = get_nni_statistics(orig_tree_ll,mean_branch_length_orig,nni_neighbors,neighbors_tmp_path,curr_run_dir, msa_path,bootstrap_tree_details_dict,garbage_dir, opt  = False)
            total_neig_ll_evaluation_time_no_opt+=neig_ll_evaluation_time_no_opt
            nni_statistics_opt,neig_ll_evaluation_time_opt = get_nni_statistics(orig_tree_ll,mean_branch_length_orig,nni_neighbors, neighbors_tmp_path, curr_run_dir,
                                                       msa_path, bootstrap_tree_details_dict, garbage_dir, opt=True)
            total_neig_ll_evaluation_time_opt+=neig_ll_evaluation_time_opt
            #nni_statistics_JC,neig_ll_evaluation_time_jc = get_nni_statistics(orig_tree_ll_JC, mean_branch_length_orig, nni_neighbors,
            #                                        neighbors_tmp_path, curr_run_dir,
            #                                        msa_path, bootstrap_tree_details_dict, garbage_dir, opt=True, simple_model= True)
            #total_neig_ll_evaluation_time_opt_JC += neig_ll_evaluation_time_jc
            statistics.update(nni_statistics_no_opt)
            statistics.update(nni_statistics_opt)
            #statistics.update(nni_statistics_JC)
            statistics.update(bootstrap_tree_details_dict)
            msa_splits = msa_splits.append(statistics, ignore_index=True)
            create_or_clean_dir(garbage_dir)

    logging.info(f"Feature extraction time = {feature_extraction_time} ")
    running_time_dict = {'extraction_of_features_time':feature_extraction_time,'total_neig_ll_evaluation_time_no_opt':total_neig_ll_evaluation_time_no_opt,'total_neig_ll_evaluation_time_opt':total_neig_ll_evaluation_time_opt}
    for col in running_time_dict:
        msa_splits[col] = running_time_dict[col]

    return msa_splits


def main():
    parser = job_parser()
    args = parser.parse_args()
    create_dir_if_not_exists(args.job_work_path)
    data = pd.read_csv(args.job_data_path, sep='\t')
    all_splits = pd.DataFrame()
    log_file_path = os.path.join(args.job_work_path, "general_features.log")
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    for true_tree_path in data['true_tree_path'].unique():
        tree_data = data.loc[data.true_tree_path == true_tree_path]
        for program in tree_data['program'].unique():
            tree_program_data = tree_data.loc[tree_data.program==program]
            for msa_path in tree_data['msa_path'].unique():
                bootstrap_tree_details = tree_program_data.loc[tree_program_data.msa_path == msa_path].head(1).squeeze()

                mle_path =  bootstrap_tree_details[get_program_default_ML_tree(program)]
                msa_splits = msa_path_edit_analysis(msa_path, args.job_work_path, mle_path, true_tree_path, program, bootstrap_tree_details.to_dict(), args.n_pars)
                all_splits = pd.concat([all_splits, msa_splits])
                all_splits.to_csv(args.job_final_output_path, sep='\t')
        raxml_data = all_splits.loc[all_splits.program=='raxml']
        raxml_data.to_csv(os.path.join(args.job_work_path,'simulations_df_raxml.tsv'), sep='\t')
        iqtree_data = all_splits.loc[all_splits.program == 'iqtree']
        iqtree_data.to_csv(os.path.join(args.job_work_path,'simulations_df_iqtree.tsv'), sep='\t')
        fasttree_data = all_splits.loc[all_splits.program == 'fasttree']
        fasttree_data.to_csv(os.path.join(args.job_work_path,'simulations_df_fasttree.tsv'), sep='\t')




if __name__ == "__main__":
    main()