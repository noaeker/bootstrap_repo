import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from ML_utils.ML_algorithms_and_hueristics import *
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_MSA_seq_names
from sklearn.metrics import balanced_accuracy_score
import os
import matplotlib.pyplot as plt
import argparse





def per_tree_analysis(test, features,model):
    balanced_accuracies = []
    for tree_id in test['tree_id'].unique():
        tree_data_test = test[test.tree_id == tree_id]
        curr_y_true = tree_data_test["true_binary_support"]
        curr_X_test = tree_data_test[[col for col in test.columns if col in features]]
        #prob_predictions = model.predict_proba(curr_X_test)[:, 1]
        predictions = model.predict(( curr_X_test))
        metrics = model_evaluation_metrics(curr_y_true, predictions, None)
        balanced_accuracies.append(metrics["balanced_accuracy_score"])
    plt.hist(balanced_accuracies)
    plt.show()

def model_pipeline(working_dir,X_train, y_train,name):
    model_path = os.path.join(working_dir,f'model_{name}')
    if  os.path.exists(model_path):
        logging.info(f"Using existing model in {model_path}")
        model = pickle.load(open(model_path, "rb"))
    else:
        model = lightgbm.LGBMClassifier(importance_type='gain').fit(X_train, y_train)
        with open(model_path,'wb') as MODEL:
            pickle.dump(model, MODEL)

    return model






def get_bootstrap_col(program):
    if program == 'raxml':
        # full_data["feature_ML_vs_pars"] = full_data["feature_parsimony_trees_binary_mean"]- full_data["feature_all_ML_ete_binary_mean"]
        bootstrap_cols = ['bootstrap_support']
    elif program == 'iqtree':
        bootstrap_cols = ['bootstrap_support', 'feature_aLRT_iqtree_support', 'feature_aBayes_iqtree_support']
    elif program == 'fasttree':
        bootstrap_cols = ['feature_standard_fasttree_boot_support', 'bootstrap_support']
    return bootstrap_cols


def bootstrap_model_pipeline(working_dir,train, y_train,test, y_test,features,bootstrap_col,groups,cpus_per_main_job,sample_frac,do_RFE , large_grid, name):
    X_train_only_boot = train[[bootstrap_col]]
    X_train_only_boot["ignore"] = 1
    X_test_only_boot = test[[bootstrap_col]]
    X_test_only_boot["ignore"] = 1
    X_train_inc_boot = train[[col for col in train.columns if col in features] + [bootstrap_col]]
    X_test_inc_boot = test[[col for col in train.columns if col in features] + [bootstrap_col]]
    model_only_boot = ML_model(X_train_only_boot, groups, y_train, n_jobs=cpus_per_main_job,
                               path=os.path.join(working_dir, f'model_only_boot'),
                               classifier=True, model='lightgbm', calibrate=True, name=name,
                               large_grid=large_grid, do_RFE=do_RFE,
                               n_cv_folds=3)
    model_inc_boot = ML_model(X_train_inc_boot, groups, y_train, n_jobs=cpus_per_main_job,
                              path=os.path.join(working_dir, f'model_inc_boot'
                              f''),
                              classifier=True, model='lightgbm', calibrate=True, name=name, large_grid=large_grid,
                              do_RFE=do_RFE,
                              n_cv_folds=3)
    data_dict_only_boot = {'test': {'X': X_test_only_boot, 'y': y_test, 'full_data': test},
                           'train': {'X': X_train_only_boot, 'y': y_train, 'full_data': train}}
    raw_boot_performance,groups_data_raw_boot = overall_model_performance_analysis(working_dir, None, data_dict_only_boot,
                                                              name="raw_only_boot",extract_predictions = False)

    only_boot_performance,groups_data_only_boot = overall_model_performance_analysis(working_dir, model_only_boot, data_dict_only_boot,
                                                               name="only_boot",extract_predictions = False)
    data_dict_inc_boot = {'test': {'X': X_test_inc_boot, 'y': y_test, 'full_data': test},
                          'train': {'X': X_train_inc_boot, 'y': y_train, 'full_data': train}}
    inc_boot_performance,groups_data_inc_boot = overall_model_performance_analysis(working_dir, model_inc_boot, data_dict_inc_boot,
                                                              name="inc_boot",extract_predictions = False)
    bootstrap_models_performance = pd.concat([raw_boot_performance, only_boot_performance, inc_boot_performance])
    bootstrap_models_performance["analysis_type"] = bootstrap_col
    return bootstrap_models_performance



def standard_model_pipeline(train,test, y_train, y_test, groups,features, cpus_per_main_job, working_dir,do_RFE , large_grid, name, validation_dict,extract_predictions):
    X_train = train[[col for col in train.columns if col in features]]
    X_test = test[[col for col in train.columns if col in features]]

    logging.info("Training ML model")
    model = ML_model(X_train, groups, y_train, n_jobs=cpus_per_main_job,
                     path=os.path.join(working_dir, f'model_stadard'),
                     classifier=True, model='lightgbm', calibrate=True, name=name, large_grid=large_grid, do_RFE=do_RFE,
                     n_cv_folds=3)

    data_dict = {'test': {'X': X_test, 'y': y_test, 'full_data': test},
                 'train': {'X': X_train, 'y': y_train, 'full_data': train}}
    for val_name in validation_dict:
        validation_data = validation_dict[val_name]
        data_dict[val_name] = {'X': validation_data[[col for col in validation_data.columns if col in features]],
                               'y': validation_data["true_binary_support"], 'full_data': validation_data}

    logging.info("Evaluating model performance")
    model_performance, group_performance = overall_model_performance_analysis(working_dir, model, data_dict,
                                                                              name=f"model_standard",
                                                                              extract_predictions=extract_predictions)
    model_performance["analysis_type"] = name
    return model_performance,group_performance



def ML_pipeline(program_data, bootstrap_cols, cpus_per_main_job, working_dir, sample_frac,subsample_train, do_RFE , large_grid, name, validation_dict, compare_to_bootstrap_models = False, extract_predictions = False):
    all_models_performance = pd.DataFrame()
    program_data = program_data.dropna(axis=1, how='all')
    program_data = program_data.dropna(axis=0)
    train, test = train_test_validation_splits(program_data, test_pct=0.3,
                                               subsample_train=subsample_train, subsample_train_frac=sample_frac)
    y_train = train["true_binary_support"]  # default_status
    y_test = test["true_binary_support"]
    groups = train["tree_id"]
    logging.info(f"Number of different trees is {len(program_data['tree_id'].unique())}")
    full_features = [col for col in program_data.columns if
                'feature' in col and 'msa_entropy' not in col and col not in bootstrap_cols and 'll_diff_norm' not in col and 'column_variance' not in col]  # +['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']
    logging.info(f"Full features are: {full_features}")
    logging.info(f"Evaluating full standard model- including nni feautres, number of features is {len(full_features)}")
    full_model_working_dir = os.path.join(working_dir,'full_model')
    create_dir_if_not_exists(full_model_working_dir)
    model_performance_full,group_performance_full = standard_model_pipeline(train,test, y_train, y_test, groups,full_features, cpus_per_main_job, full_model_working_dir,do_RFE , large_grid, name, validation_dict,extract_predictions)
    model_performance_full["analysis_type"] = "full_standard"
    all_models_performance = pd.concat([all_models_performance, model_performance_full])
    nni_cols = ['feature_min_ll_diff','feature_max_ll_diff', 'feature_column_variance',
                              'feature_min_ll_diff_norm','feature_max_ll_diff_norm','feature_column_variance_norm']

    fast_features = [col for col in full_features if col not in nni_cols]  # +['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']

    logging.info(f"Evaluating fast standard model- no nni feautres, number of features is {len(fast_features)}")
    fast_model_working_dir = os.path.join(working_dir, 'fast_model')
    create_dir_if_not_exists(fast_model_working_dir)
    model_performance_fast, group_performance_fast = standard_model_pipeline(train,test, y_train, y_test, groups,fast_features, cpus_per_main_job, fast_model_working_dir,do_RFE , large_grid, name, validation_dict,extract_predictions)
    model_performance_fast["analysis_type"] = "fast_standard"
    all_models_performance = pd.concat([all_models_performance, model_performance_fast])


    if compare_to_bootstrap_models:
        logging.info("Comparing to bootstrap models")
        for bootstrap_col in bootstrap_cols:
            bootstrap_working_dir = os.path.join(working_dir,bootstrap_col)
            create_dir_if_not_exists(bootstrap_working_dir)
            bootstrap_models_performance = bootstrap_model_pipeline(bootstrap_working_dir,train, y_train,test, y_test,full_features,bootstrap_col,groups,cpus_per_main_job,sample_frac,do_RFE , large_grid, name = name+bootstrap_col)
            all_models_performance = pd.concat([all_models_performance, bootstrap_models_performance])

    all_models_performance["sample_frac"] = sample_frac
    group_performance_full["sample_frac"] = sample_frac
    return all_models_performance,group_performance_full





def transform_data(df):
    df['true_binary_support'] = df['true_support'] == 1
    df['feature_msa_n_seq'] = df['feature_n_unique_seq']
    df.drop(columns = ['feature_n_unique_seq'], inplace= True)
    #+[col for col in df.columns if 'msa_entropy' in col]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type = str, default = os.getcwd())
    parser.add_argument('--RFE', action='store_true', default= False)
    parser.add_argument('--full_grid', action='store_true', default=False)
    parser.add_argument('--cpus_per_main_job', type = int, default=1)
    parser.add_argument('--sample_fracs', type = str, default='0.25_0.5_1')
    parser.add_argument('--inc_sample_fracs', action='store_true', default=False)
    parser.add_argument('--main_data_folder',type = str, default = '/Users/noa/Workspace/bootstrap_results/remote_results/full_data')
    parser.add_argument('--validation_data_folder', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/remote_results/validation_data')
    args = parser.parse_args()
    log_file_path = os.path.join(args.working_dir, "ML.log")
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    for program in ['fasttree','iqtree','raxml']: #,'fasttree','raxml'
        logging.info(f"Program = {program}")
        program_data = pd.read_csv(os.path.join(args.main_data_folder,f'simulations_df_{program}.tsv'),sep='\t')
        transform_data(program_data)
        program_validation_data = pd.read_csv(os.path.join(args.validation_data_folder,f'simulations_df_{program}.tsv'),sep='\t')
        transform_data(program_validation_data)
        working_dir = os.path.join(args.working_dir, program)
        create_dir_if_not_exists(working_dir)
        bootstrap_cols = get_bootstrap_col(program)
        program_data = program_data.dropna(axis=1, how='all')
        program_data = program_data.dropna(axis=0)
        sample_fracs = [float(frac) for frac in (args.sample_fracs).split('_')]
        all_model_merics = pd.DataFrame()
        validation_dict = {}
        for model_mode in np.unique(program_validation_data["model_mode"]):
            validation_dict[f'val_{model_mode}'] = program_validation_data.loc[program_validation_data.model_mode==model_mode].copy()
        if args.inc_sample_fracs:
            for sample_frac in sample_fracs:
                logging.info(f"\n#Sample frac = {sample_frac}")
                sample_frac_working_dir = os.path.join(working_dir,f"frac_{sample_frac}")
                create_dir_if_not_exists(sample_frac_working_dir)
                curr_model_metrics, groups_analysis = ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job, sample_frac_working_dir, sample_frac,compare_to_bootstrap_models= False,subsample_train = True,do_RFE = args.RFE, large_grid = False, name = f"frac_{sample_frac}", validation_dict = validation_dict)
                all_model_merics = pd.concat([all_model_merics,curr_model_metrics])
        all_model_merics.to_csv(os.path.join(working_dir, 'all_models_performance.tsv'), sep=CSV_SEP)
        logging.info(f"Generating optimized final model")
        final_model_working_dir = os.path.join(working_dir, f"final_model")
        create_dir_if_not_exists(final_model_working_dir)
        final_model_metrics,groups_analysis = ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job, final_model_working_dir, sample_frac = -1,subsample_train = False, do_RFE=args.RFE,
                    large_grid=args.full_grid, name=f"final_model", validation_dict = validation_dict, compare_to_bootstrap_models= True, extract_predictions = True)
        final_model_metrics.to_csv(os.path.join(working_dir, 'final_model_performance.tsv'), sep=CSV_SEP)
        groups_analysis.to_csv(os.path.join(working_dir, 'groups_performance.tsv'), sep=CSV_SEP)

    #     print(test_metrics)

if __name__ == "__main__":
    main()