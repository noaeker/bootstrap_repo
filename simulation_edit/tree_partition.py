from side_code.basic_trees_manipulation import *
import dendropy
import numpy as np
import pandas as pd
from side_code.basic_trees_manipulation import *
from dendropy.calculate import treecompare
from side_code.code_submission import execute_command_and_write_to_log
from sklearn.metrics import silhouette_samples, silhouette_score
#def get_partition_score(tree_obj):
#    for node in tree_obj:

def get_all_pairwise_distances_mat(taxa, pdc):
    pairwise_distances = []
    for i, t1 in enumerate(taxa[:-1]):
       for t2 in taxa[i + 1:]:
           pairwise_distances.append(pdc(t1, t2))
    return pairwise_distances

def get_tree_obj(newick, taxa):
    tree = dendropy.Tree.get(data=newick, schema='newick', rooting='force-unrooted',
                             taxon_namespace=taxa)
    #print(tree.as_ascii_plot())
    tree.encode_bipartitions()
    #for edge in tree.postorder_edge_iter():
    #    edge.length = 1
    return tree


def main():
    data = pd.read_csv("/Users/noa/Workspace/simulations_results/total_simulations.tsv", sep = '\t', nrows = 1000)
    data = data.loc[data.final_ll>data.LL_opt+1].head(1)
    with open("booster.nw") as TEST:
        nw = TEST.read()
    #taxa = dendropy.TaxonNamespace()
    #test_tree = get_tree_obj(nw, taxa)
    #for edge in test_tree.postorder_edge_iter():
    #    bipartition_str = np.array(list(str(edge.bipartition)))
    #    curr_taxa = np.array(taxa[::-1])[np.where(bipartition_str=='1')]
    #    support = edge.head_node.label
    #    print(support)
    #    #trimmed_tree =


    #     curr_taxa_distances = get_all_pairwise_distances_mat(curr_taxa, true_pdc)




    true_tree_nk = np.max(data['tree_str'])
    #
    # true_tree = get_tree_obj(true_tree_nk, taxa)
    # curr_true_tree_path = "curr_true_tree.tree"
    # second_tree_path = "second_tree.tree"
    # #print(true_tree.as_ascii_plot())
    # true_pdc = true_tree.phylogenetic_distance_matrix()
    # second_tree_nk = np.max(data['final_tree_topology'])
    # second_tree = get_tree_obj(second_tree_nk, taxa)
    # second_pdc = second_tree.phylogenetic_distance_matrix()
    # print(treecompare.robinson_foulds_distance(second_tree, true_tree))
    #
    # with open(curr_true_tree_path,'w') as TRUE:
    #     TRUE.write(true_tree_nk)
    # with open(second_tree_path,'w') as SECOND_TREE:
    #     SECOND_TREE.write(second_tree_nk)
    # cmd = f"{BOOSTER_EXE} -a tbe -i {curr_true_tree_path} -b {second_tree_path} -@ 1 -o booster.nw -r booster.log -S"
    # execute_command_and_write_to_log(cmd)
    # print(cmd)



    # for bipartition in true_tree.bipartition_encoding:
    #     bipartition_str = np.array(list(str(bipartition)))
    #     curr_taxa = np.array(taxa[::-1])[np.where(bipartition_str=='1')]
    #     curr_taxa_distances = get_all_pairwise_distances_mat(curr_taxa, true_pdc)
    #
    #
    #     new_distances = get_all_pairwise_distances_mat(curr_taxa, second_pdc)
    #     if curr_taxa_distances!= new_distances:
    #         print([s.label for s in curr_taxa])
    #         print(curr_taxa_distances)
    #         print(new_distances)






if __name__ == "__main__":
    main()
