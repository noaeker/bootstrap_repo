**In this work, we propose a machine-learning based approach to eatimate branch support in phylogenies**

1. Our work is based on simulations of many MSAs, and applying tree-searches using several phylogenetic softwares. The simulations generation is implemented in the 'simulations_generation_folder'.
Simulations are performed based on empirical trees.
3. After running the simulaions and tree-searches, we extract features from the inferred ML trees, in the 'simulation_edit' folder.
4. Finally, we use the extracted features to predict branch support values, as implemented in the 'ML_pipeline' folder.

The trained models and the empirical trees and MSAs used for training can be downloaded from Figshare DOI 10.6084/m9.figshare.25050554.  



The "predict_branch_supports.py" file in the "user_code" directory contains a user-friendly code designed for utilizing our model to predict branch supports.  
**Input**:  
--working_dir  
--mle_tree_path, path of maximum likelihood tree   
--all_mles_tree_path, path of a set of obtained maximum likelihood tree (such as the files obtained by RAxML-NG)      
--msa_path, path of the multiple sequence alignemnt  
--model, DNA subtitution model used in tree-search  
--trained_ML_model_path, path of the trained ML model   
--raxml_ng_path, path of RAxML-NG program, used for features which rely on log-likelihood estimation.(Kozlov et al., 2019)   
--mad_path, path of RAxML-NG program (Tria et al., 2017)   
--booster_path, path of transfer distance program  (Lemoine et al., 2018)    


Kozlov,A.M. et al. (2019) RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference. Bioinformatics, 35, 4453–4455.    
Tria,F.D.K. et al. (2017) Phylogenetic rooting using minimal ancestor deviation. Nat. Ecol. Evol., 1, 193.    
Lemoine,F. et al. (2018) Renewing Felsenstein’s phylogenetic bootstrap in the era of big data. Nat. 2018 5567702, 556, 452–456.    



