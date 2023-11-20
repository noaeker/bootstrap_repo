import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from ML_utils.ML_algorithms_and_hueristics import *
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.combine_current_results import unify_results_across_jobs
from side_code.MSA_manipulation import get_MSA_seq_names
from sklearn.metrics import balanced_accuracy_score
import os
import matplotlib.pyplot as plt
import argparse


def per_tree_analysis(test, features, model):
    balanced_accuracies = []
    for tree_id in test['tree_id'].unique():
        tree_data_test = test[test.tree_id == tree_id]
        curr_y_true = tree_data_test["true_binary_support"]
        curr_X_test = tree_data_test[[col for col in test.columns if col in features]]
        # prob_predictions = model.predict_proba(curr_X_test)[:, 1]
        predictions = model.predict((curr_X_test))
        metrics = model_evaluation_metrics(curr_y_true, None)
        balanced_accuracies.append(metrics["balanced_accuracy_score"])
    plt.hist(balanced_accuracies)
    plt.show()


def model_pipeline(working_dir, X_train, y_train, name):
    model_path = os.path.join(working_dir, f'model_{name}')
    if os.path.exists(model_path):
        logging.info(f"Using existing model in {model_path}")
        model = pickle.load(open(model_path, "rb"))
    else:
        model = lightgbm.LGBMClassifier(importance_type='gain').fit(X_train, y_train)
        with open(model_path, 'wb') as MODEL:
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


def generate_data_dict_from_column_subset(data_dict, column_subset):
    res = {}
    for dataset in data_dict:
        data = data_dict[dataset]
        X = data[column_subset]
        if len(column_subset) == 1:
            X["ignore"] = 1
        res[dataset] = {'X': X, 'y': data['true_binary_support'], 'full_data': data}
    return res


def bootstrap_model_pipeline(working_dir, train, test, validation_dict, features, bootstrap_col, groups,
                             cpus_per_main_job, sample_frac, do_RFE, large_grid, name):
    data_dict = {'test': test, 'train': train}
    data_dict.update(validation_dict)
    only_boot_dict = generate_data_dict_from_column_subset(data_dict, column_subset=[bootstrap_col])

    inc_boot_dict = generate_data_dict_from_column_subset(data_dict, column_subset=[col for col in train.columns if
                                                                                    col in features] + [bootstrap_col])

    model_only_boot = ML_model(only_boot_dict["train"]["X"], groups, only_boot_dict["train"]["y"],
                               n_jobs=cpus_per_main_job,
                               path=os.path.join(working_dir, f'model_only_boot'),
                               classifier=True, model='lightgbm', calibrate=True, name=name,
                               large_grid=large_grid, do_RFE=do_RFE,
                               n_cv_folds=3)
    model_inc_boot = ML_model(inc_boot_dict["train"]["X"], groups, inc_boot_dict["train"]["y"],
                              n_jobs=cpus_per_main_job,
                              path=os.path.join(working_dir, f'model_inc_boot'
                              f''),
                              classifier=True, model='lightgbm', calibrate=True, name=name, large_grid=large_grid,
                              do_RFE=do_RFE,
                              n_cv_folds=3)

    raw_boot_performance, groups_data_raw_boot,test_predictions_raw_boot = overall_model_performance_analysis(working_dir, None, only_boot_dict,
                                                                                    name="raw_only_boot",
                                                                                    extract_predictions=False)

    only_boot_performance, groups_data_only_boot,test_predictions_only_boot = overall_model_performance_analysis(working_dir, model_only_boot,
                                                                                      only_boot_dict,
                                                                                      name="only_boot",
                                                                                      extract_predictions=False)
    inc_boot_performance, groups_data_inc_boot,test_predictions_inc_boot = overall_model_performance_analysis(working_dir, model_inc_boot,
                                                                                    inc_boot_dict,
                                                                                    name="inc_boot",
                                                                                    extract_predictions=False)
    bootstrap_models_performance = pd.concat([raw_boot_performance, only_boot_performance, inc_boot_performance])
    bootstrap_models_performance["analysis_type"] = bootstrap_col

    test_performance_dict = {f'{bootstrap_col}_only_boot' : test_predictions_only_boot,f'{bootstrap_col}_raw_boot' : test_predictions_raw_boot,f'{bootstrap_col}_inc_boot' : test_predictions_inc_boot}
    return bootstrap_models_performance,test_performance_dict


def standard_model_pipeline(train, test, validation_dict, groups, features, cpus_per_main_job, working_dir, do_RFE,
                            large_grid, name, extract_predictions):
    all_datasets = {'test': test, 'train': train}
    all_datasets.update(validation_dict)
    datasets_dict = generate_data_dict_from_column_subset(all_datasets, column_subset=[col for col in train.columns if
                                                                                       col in features])

    logging.info("Training ML model")
    model = ML_model(datasets_dict["train"]["X"], groups, datasets_dict["train"]["y"], n_jobs=cpus_per_main_job,
                     path=os.path.join(working_dir, f'model_stadard'),
                     classifier=True, model='lightgbm', calibrate=True, name=name, large_grid=large_grid, do_RFE=do_RFE,
                     n_cv_folds=3)

    logging.info("Evaluating model performance")
    model_performance, group_performance, test_predictions = overall_model_performance_analysis(working_dir, model, datasets_dict,
                                                                              name=f"model_standard",
                                                                              extract_predictions=extract_predictions)
    model_performance["analysis_type"] = name
    return model_performance, group_performance, test_predictions


def ML_pipeline(program_data, bootstrap_cols, cpus_per_main_job, working_dir, sample_frac, subsample_train, do_RFE,
                large_grid, name, validation_dict, compare_to_bootstrap_models=False, extract_predictions=False):
    all_models_performance = pd.DataFrame()
    all_test_perdictions = {}
    program_data = program_data.dropna(axis=1, how='all')
    program_data = program_data.dropna(axis=0)
    train, test = train_test_validation_splits(program_data, test_pct=0.3,
                                               subsample_train=subsample_train, subsample_train_frac=sample_frac)
    groups = train["tree_id"]
    logging.info(f"Number of different trees is {len(program_data['tree_id'].unique())}")
    full_features = [col for col in program_data.columns if
                     'feature' in col and col not in bootstrap_cols]  # +['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']
    logging.info(f"Full features are: {full_features}")
    logging.info(f"Evaluating full standard model- including nni feautres, number of features is {len(full_features)}")
    full_model_working_dir = os.path.join(working_dir, 'full_model')
    create_dir_if_not_exists(full_model_working_dir)
    model_performance_full, group_performance_full, test_predictions_full = standard_model_pipeline(train, test, validation_dict, groups,
                                                                             full_features, cpus_per_main_job,
                                                                             full_model_working_dir, do_RFE, large_grid,
                                                                             name, extract_predictions)
    all_test_perdictions["full_standard"] = test_predictions_full
    model_performance_full["analysis_type"] = "full_standard"
    all_models_performance = pd.concat([all_models_performance, model_performance_full])
    nni_cols = ['feature_min_ll_diff_opt=True_model_False', 'feature_max_ll_diff_opt=True_model_False']

    fast_features = [col for col in full_features if
                     col not in nni_cols]  # +['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']

    logging.info(f"Evaluating fast standard model- no nni feautres, number of features is {len(fast_features)}")
    fast_model_working_dir = os.path.join(working_dir, 'fast_model')
    create_dir_if_not_exists(fast_model_working_dir)
    model_performance_fast, group_performance_fast,test_predictions_fast = standard_model_pipeline(train, test, validation_dict, groups,
                                                                             fast_features, cpus_per_main_job,
                                                                             fast_model_working_dir, do_RFE, large_grid,
                                                                             name, extract_predictions)
    all_test_perdictions["fast_standard"] = test_predictions_full
    model_performance_fast["analysis_type"] = "fast_standard"
    all_models_performance = pd.concat([all_models_performance, model_performance_fast])

    if compare_to_bootstrap_models:
        logging.info("Comparing to bootstrap models")
        for bootstrap_col in bootstrap_cols:
            logging.info(f"Bootstrap col {bootstrap_col}")
            bootstrap_working_dir = os.path.join(working_dir, bootstrap_col)
            create_dir_if_not_exists(bootstrap_working_dir)
            bootstrap_models_performance, test_performance_dict = bootstrap_model_pipeline(bootstrap_working_dir, train, test, validation_dict,
                                                                    full_features, bootstrap_col, groups,
                                                                    cpus_per_main_job, sample_frac, do_RFE, large_grid,
                                                                    name=name + bootstrap_col)
            all_test_perdictions.update(test_performance_dict)
            all_models_performance = pd.concat([all_models_performance, bootstrap_models_performance])

    all_models_performance["sample_frac"] = sample_frac
    group_performance_full["sample_frac"] = sample_frac
    if sample_frac==-1:
        enriched_test = test.copy().reset_index()
        test_performance_df = pd.DataFrame(test_performance_dict).reset_index()
        test_with_predictions = pd.concat([enriched_test,test_performance_df],axis=1)
    return all_models_performance, group_performance_full,test_with_predictions


def get_n_free_parameters(x):
    if x == 'GTR':
        n = 8
    elif x == 'K80':
        n = 2
    elif x == 'JC':
        n = 1
    elif x == 'HKY':
        n = 2
    elif x == 'TVM':
        n = 7
    elif x == 'SYM':
        n = 5
    elif x == 'F81':
        n = 3
    elif x == 'TIM':
        n = 6
    elif x == 'F84':
        n = 4
    elif x == 'TPM3':
        n = 3
    else:
        n = -1
    return n


def transform_data(df):
    df['true_binary_support'] = df['true_support'] == 1
    #df['feature_msa_n_seq'] = df['feature_n_unique_seq']
    df.drop(columns=['feature_n_unique_seq'], inplace=True)
    #df['feature_G'] = df['model_short'].str.contains('+G', regex=False)
    #df['feature_I'] = df['model_short'].str.contains('+I', regex=False)
    #df['feature_F'] = df['model_short'].str.contains('+F', regex=False)
    if 'model_short' not in df.columns:
        df['model_short'] = df['tree_search_model']
    #df['model_short'] = df['model_short'].apply(lambda x: x.split('+')[0])
    #df['feature_free_parameters'] = df['model_short'].apply(lambda x: get_n_free_parameters(x))
    df = df[[col for col in df.columns if
             'msa_entropy' not in col and 'extraction_of_features_time' not in col and 'feature_abayes_opt' not in col and 'column_variance' not in col and 'feature_min_mean_branch_length' not in col and 'feature_max_mean_branch_length' and 'll_diff_norm' not in col ]]
    # df = pd.get_dummies(df,prefix='feature_model_',columns=['model_short']) #
    return df
    # +[col for col in df.columns if 'msa_entropy' in col]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--RFE', action='store_true', default=False)
    parser.add_argument('--full_grid', action='store_true', default=False)
    parser.add_argument('--programs', type=str, default='iqtree_raxml_fasttree')
    parser.add_argument('--cpus_per_main_job', type=int, default=1)
    parser.add_argument('--sample_fracs', type=str, default='0.25_0.5_1')
    parser.add_argument('--inc_sample_fracs', action='store_true', default=False)
    parser.add_argument('--reunite_val_data', action='store_true', default=False)
    parser.add_argument('--use_val_data', action='store_true', default=False)
    parser.add_argument('--reunite_training_data', action='store_true', default=False)
    parser.add_argument('--main_data_folder', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/remote_results/full_data')
    parser.add_argument('--validation_data_folder', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/remote_results/validation_data')
    parser.add_argument('--sample_val', action='store_true', default=False)
    args = parser.parse_args()
    log_file_path = os.path.join(args.working_dir, "ML.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    create_dir_if_not_exists(args.working_dir)
    for program in args.programs.split('_'):
        logging.info(f"Program = {program}")
        training_data_path = os.path.join(args.main_data_folder, f'simulations_df_{program}.tsv')
        if args.reunite_training_data or not os.path.exists(training_data_path):
            logging.info(f"Re-uniting training data and saving to {training_data_path}")
            program_data = unify_results_across_jobs(args.main_data_folder,
                                                     name=f'simulations_df_{program}', n_jobs=1000)
        else:
            logging.info(f"Using existing training data in {training_data_path} ")
            program_data = pd.read_csv(training_data_path, sep='\t')
        program_data = transform_data(program_data)
        validation_dict = {}
        validation_data_path = os.path.join(args.validation_data_folder, f'simulations_df_{program}.tsv')
        if args.use_val_data:
            logging.info("Using validation data")
            if (args.reunite_val_data or not os.path.exists(validation_data_path)):
                logging.info(f"Re-uniting validation data and saving to {validation_data_path}")
                program_validation_data = unify_results_across_jobs(args.validation_data_folder,
                                                                    name=f'simulations_df_{program}', n_jobs=1000)
            else:
                logging.info("Using existing validation data")
                program_validation_data = pd.read_csv(validation_data_path, sep='\t')
            program_validation_data = transform_data(program_validation_data)
            for model_mode in np.unique(program_validation_data["model_mode"]):
                validation_dict[f'val_{model_mode}'] = program_validation_data.loc[
                    program_validation_data.model_mode == model_mode].copy()
        working_dir = os.path.join(args.working_dir, program)
        create_dir_if_not_exists(working_dir)
        bootstrap_cols = get_bootstrap_col(program)
        program_data = program_data.dropna(axis=1, how='all')
        program_data = program_data.dropna(axis=0)
        sample_fracs = [float(frac) for frac in (args.sample_fracs).split('_')]
        all_model_merics = pd.DataFrame()
        if args.inc_sample_fracs:
            for sample_frac in sample_fracs:
                logging.info(f"\n#Sample frac = {sample_frac}")
                sample_frac_working_dir = os.path.join(working_dir, f"frac_{sample_frac}")
                create_dir_if_not_exists(sample_frac_working_dir)
                curr_model_metrics, groups_analysis = ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job,
                                                                  sample_frac_working_dir, sample_frac,
                                                                  compare_to_bootstrap_models=False,
                                                                  subsample_train=True, do_RFE=args.RFE,
                                                                  large_grid=False, name=f"frac_{sample_frac}",
                                                                  validation_dict=validation_dict)
                all_model_merics = pd.concat([all_model_merics, curr_model_metrics])
        all_model_merics.to_csv(os.path.join(working_dir, 'all_models_performance.tsv'), sep=CSV_SEP)
        logging.info(f"Generating optimized final model")
        final_model_working_dir = os.path.join(working_dir, f"final_model")
        create_dir_if_not_exists(final_model_working_dir)
        final_model_metrics, groups_analysis,test_with_predictions = ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job,
                                                           final_model_working_dir, sample_frac=-1,
                                                           subsample_train=False, do_RFE=args.RFE,
                                                           large_grid=args.full_grid, name=f"final_model",
                                                           validation_dict=validation_dict,
                                                           compare_to_bootstrap_models=True, extract_predictions=True)
        final_model_metrics.to_csv(os.path.join(working_dir, 'final_model_performance.tsv'), sep=CSV_SEP)
        groups_analysis.to_csv(os.path.join(working_dir, 'groups_performance.tsv'), sep=CSV_SEP)
        test_with_predictions.to_csv(os.path.join(working_dir, 'test_with_predictions.tsv'), sep=CSV_SEP)

    #     print(test_metrics)


if __name__ == "__main__":
    main()
