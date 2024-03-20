import pandas as pd
from ete3 import Tree
from ete3 import PhyloTree
import os
import matplotlib.pyplot as plt



def main():
    iqtree_rpl16_df = pd.read_csv('/Users/noa/Workspace/Doro_real_iqtree.csv')
    iqtree_rpl16_df = iqtree_rpl16_df.loc[iqtree_rpl16_df["study_name"]=='rpl16b']
    iqtree_rpl_16_tree = PhyloTree  ('/Users/noa/Workspace/rpl16b.nw', format=1)
    iqtree_rpl16_df["diff_vs_aLRT"] = abs(iqtree_rpl16_df["predicted_bootstrap_score_nni"]-iqtree_rpl16_df['aLRT_iqtree'])
    iqtree_rpl16_df["diff_vs_aBayes"] = abs(iqtree_rpl16_df["predicted_bootstrap_score_nni"]-iqtree_rpl16_df['aBayes_iqtree'])

    #alrt_diff_extreme_nodes = iqtree_rpl16_df.sort_values("diff_vs_aLRT", ascending= False).head(5)
    #print(f"aLRT diffs:\n {alrt_diff_extreme_nodes['node_name']}")
    #aBayes_diff_extreme_nodes = iqtree_rpl16_df.sort_values("diff_vs_aBayes", ascending=False).head(5)
    #print(f"aBayes diffs\n: {aBayes_diff_extreme_nodes['node_name'] }")
    #all_extremes = pd.concat([alrt_diff_extreme_nodes,aBayes_diff_extreme_nodes])

    iqtree_rpl16_df.sort_values("diff_vs_aLRT", ascending= False)[["node_name","predicted_bootstrap_score_nni","aLRT_iqtree","aBayes_iqtree"]].to_csv("rpl16b_bootstrap_scores.csv")
    for i, row in iqtree_rpl16_df.iterrows():
        (iqtree_rpl_16_tree & row["node_name"]).support = row["predicted_bootstrap_score_nni"]
    iqtree_rpl_16_tree.write(format=1, outfile="rpl16b_ML.nw")
    for i, row in iqtree_rpl16_df.iterrows():
        (iqtree_rpl_16_tree & row["node_name"]).support = row['aBayes_iqtree']
    iqtree_rpl_16_tree.write(format=1, outfile="rpl16b_aBayes.nw")
    for i, row in iqtree_rpl16_df.iterrows():
        (iqtree_rpl_16_tree & row["node_name"]).support = row['aLRT_iqtree']
        #print(row['aLRT_iqtree'])
    iqtree_rpl_16_tree.write(format=1, outfile="rpl16b_aLRT.nw")


if __name__ == "__main__":
    main()



# for i, row in combined_df.iterrows():
#    (ML_tree & row["node_name"]).support = row["predicted_bootstrap_score_nni"]
# ML_tree.write(format=0, outfile=os.path.join(args.raw_results_folder, "ML_tree_with_predictions.nw"))
