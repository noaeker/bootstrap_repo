import sys
import random
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from ML_utils.ML_algorithms_and_hueristics import *
from ML_pipeline.ML_argparser import main_parser
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.combine_current_results import unify_results_across_jobs
import os
import matplotlib.pyplot as plt
from random import sample
import argparse
from functools import reduce



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
        bootstrap_cols = ['bootstrap_support']
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


def bootstrap_model_pipeline(working_dir, train, test, validation_dict,ML_model, features, bootstrap_col, groups,
                             cpus_per_main_job, sample_frac, do_RFE, large_grid, name):
    data_dict = {'test': test, 'train': train}
    data_dict.update(validation_dict)
    only_boot_dict = generate_data_dict_from_column_subset(data_dict, column_subset=[bootstrap_col])

    inc_boot_dict = generate_data_dict_from_column_subset(data_dict, column_subset=[col for col in train.columns if
                                                                                    col in features] + [bootstrap_col])

    model_only_boot = ML_training(only_boot_dict["train"]["X"], groups, only_boot_dict["train"]["y"],
                               n_jobs=cpus_per_main_job,
                               path=os.path.join(working_dir, f'model_only_boot'),
                               classifier=True, model=ML_model, calibrate=True, name=name,
                               large_grid=large_grid, do_RFE=do_RFE,
                               n_cv_folds=3)
    model_inc_boot = ML_training(inc_boot_dict["train"]["X"], groups, inc_boot_dict["train"]["y"],
                              n_jobs=cpus_per_main_job,
                              path=os.path.join(working_dir, f'model_inc_boot'
                              f''),
                              classifier=True, model=ML_model, calibrate=True, name=name, large_grid=large_grid,
                              do_RFE=do_RFE,
                              n_cv_folds=3)

    raw_boot_performance, groups_data_raw_boot,predictions_raw_boot = overall_model_performance_analysis(working_dir,
                                                                                                         None,
                                                                                                         only_boot_dict,
                                                                                                         ML_model,
                                                                                                         name="raw_only_boot")

    only_boot_performance, groups_data_only_boot,predictions_only_boot = overall_model_performance_analysis(working_dir,
                                                                                                            model_only_boot,
                                                                                                            only_boot_dict,
                                                                                                            ML_model,
                                                                                                            name="only_boot")
    inc_boot_performance, groups_data_inc_boot,predictions_inc_boot = overall_model_performance_analysis(working_dir,
                                                                                                         model_inc_boot,
                                                                                                         inc_boot_dict,
                                                                                                         ML_model,
                                                                                                         name="inc_boot")
    bootstrap_models_performance = pd.concat([raw_boot_performance, only_boot_performance, inc_boot_performance])
    bootstrap_models_performance["analysis_type"] = bootstrap_col

    performance_dict = {f'predictions_{bootstrap_col}_only_boot' : predictions_only_boot,f'predictions_{bootstrap_col}_raw_boot' : predictions_raw_boot,f'predictions_{bootstrap_col}_inc_boot' : predictions_inc_boot}
    return bootstrap_models_performance,performance_dict


def standard_model_pipeline(train, test, validation_dict,ML_model, groups, features, cpus_per_main_job, working_dir, do_RFE,
                            large_grid, name, extract_predictions):
    all_datasets = {'test': test, 'train': train}
    all_datasets.update(validation_dict)
    datasets_dict = generate_data_dict_from_column_subset(all_datasets, column_subset=[col for col in train.columns if
                                                                                       col in features])

    logging.info("Training ML model")
    model = ML_training(datasets_dict["train"]["X"], groups, datasets_dict["train"]["y"], n_jobs=cpus_per_main_job,
                     path=os.path.join(working_dir, f'model_stadard'),
                     classifier=True, model=ML_model, calibrate=True, name=name, large_grid=large_grid, do_RFE=do_RFE,
                     n_cv_folds=3)

    logging.info("Evaluating model performance")
    model_performance, group_performance, model_predictions = overall_model_performance_analysis(working_dir, model,
                                                                                                 datasets_dict,ML_model,
                                                                                                 name=f"model_standard")
    model_performance["analysis_type"] = name
    return model_performance, group_performance, model_predictions



def generate_enriched_datasets(working_dir,all_perdictions,train,test,validation_dict):
    validation_dict.update({'train':train,'test':test})
    for dataset in validation_dict:
        dataset_enriched = validation_dict[dataset].copy()
        for prediction in all_perdictions:
            dataset_enriched[prediction] = all_perdictions[prediction][dataset]
        dataset_enriched.to_csv(os.path.join(working_dir,f'{dataset}.tsv'),sep='\t')


def ML_pipeline(program_data, bootstrap_cols, cpus_per_main_job, working_dir, sample_frac, subsample_train, do_RFE,
                large_grid, name, validation_dict, ML_model, compare_to_bootstrap_models=False, extract_predictions=False):
    all_models_performance = pd.DataFrame()
    all_perdictions = {}
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
    model_performance_full, group_performance_full, predictions_full = standard_model_pipeline(train, test, validation_dict, ML_model, groups,
                                                                             full_features, cpus_per_main_job,
                                                                             full_model_working_dir, do_RFE, large_grid,
                                                                             name, extract_predictions)
    all_perdictions["predictions_full_standard"] = predictions_full
    model_performance_full["analysis_type"] = "full_standard"
    all_models_performance = pd.concat([all_models_performance, model_performance_full])
    nni_cols = ['feature_min_ll_diff_opt=True_model_False', 'feature_max_ll_diff_opt=True_model_False']

    fast_features = [col for col in full_features if
                     col not in nni_cols]  # +['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']

    logging.info(f"Evaluating fast standard model- no nni feautres, number of features is {len(fast_features)}")
    fast_model_working_dir = os.path.join(working_dir, 'fast_model')
    create_dir_if_not_exists(fast_model_working_dir)
    model_performance_fast, group_performance_fast,predictions_fast = standard_model_pipeline(train, test, validation_dict, ML_model, groups,
                                                                             fast_features, cpus_per_main_job,
                                                                             fast_model_working_dir, do_RFE, large_grid,
                                                                             name, extract_predictions)
    all_perdictions["predictions_fast_standard"] = predictions_fast
    model_performance_fast["analysis_type"] = "fast_standard"
    all_models_performance = pd.concat([all_models_performance, model_performance_fast])

    if compare_to_bootstrap_models:
        logging.info("Comparing to bootstrap models")
        for bootstrap_col in bootstrap_cols:
            logging.info(f"Bootstrap col {bootstrap_col}")
            bootstrap_working_dir = os.path.join(working_dir, bootstrap_col)
            create_dir_if_not_exists(bootstrap_working_dir)
            bootstrap_models_performance, bootstrap_performance_dict = bootstrap_model_pipeline(bootstrap_working_dir, train, test, validation_dict, ML_model,
                                                                    full_features, bootstrap_col, groups,
                                                                    cpus_per_main_job, sample_frac, do_RFE, large_grid,
                                                                    name=name + bootstrap_col)
            all_perdictions.update(bootstrap_performance_dict)
            all_models_performance = pd.concat([all_models_performance, bootstrap_models_performance])

    all_models_performance["sample_frac"] = sample_frac
    group_performance_full["sample_frac"] = sample_frac
    if extract_predictions:
        generate_enriched_datasets(working_dir, all_perdictions, train, test, validation_dict)
    return all_models_performance, group_performance_full






def transform_data(df, program):
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
             'msa_entropy' not in col and 'extraction_of_features_time' not in col and 'feature_abayes_opt' not in col and 'column_variance' not in col and 'feature_min_mean_branch_length' not in col and 'feature_max_mean_branch_length' not in col and 'll_diff_norm' not in col and 'opt=False_model_False' not in col ]]
    # df = pd.get_dummies(df,prefix='feature_model_',columns=['model_short']) #
    if program=='raxml':
        df['bootstrap_support'] =df['bootstrap_support']/100
    elif program=='iqtree':
        df['bootstrap_support'] = df['bootstrap_support']/100
        df['feature_aLRT_iqtree_support']= np.minimum(df['feature_aLRT_iqtree_support'],1)
        df['feature_aBayes_iqtree_support'] = np.minimum(df['feature_aBayes_iqtree_support'],1)
    elif program=='fasttree':
        df['bootstrap_support'] = np.minimum(df['bootstrap_support'], 1)
    return df
    # +[col for col in df.columns if 'msa_entropy' in col]


def generate_data_dict_per_program(programs, folder, n_samp):
    data_per_program = {}
    for program in programs:
        data_path = os.path.join(folder, f'simulations_df_{program}.tsv')
        if not os.path.exists(data_path):
            logging.info(f"Re-uniting data and saving to {data_path}")
            program_data = unify_results_across_jobs(folder,
                                                     name=f'simulations_df_{program}', n_jobs=1000)
        else:
            logging.info(f"Using existing training data in {data_path} ")
            program_data = pd.read_csv(data_path, sep='\t')
        program_data = program_data.dropna(axis=1, how='all')
        program_data = program_data.dropna(axis=0, how='all')
        program_data= transform_data(program_data, program)
        data_per_program[program] = program_data
    data_per_program = retain_only_common_tree_ids(data_per_program, n_samp)
    return data_per_program


def retain_only_common_tree_ids(data_per_program, n_samp):
    tree_ids_list = []
    for program in data_per_program:
        tree_ids = np.unique(data_per_program[program]['tree_id'])
        tree_ids_list.append(tree_ids)
    common_tree_ids = list(reduce(np.intersect1d, tree_ids_list))
    random.seed(42)
    common_tree_ids = random.sample(common_tree_ids, n_samp)
    logging.info(f"Number of common tree IDS is {len(common_tree_ids)}")
    for program in data_per_program:
        program_data = data_per_program[program]
        program_data = program_data.loc[program_data.tree_id.isin(common_tree_ids)]
        data_per_program[program] = program_data
    return data_per_program



def main():
    parser = main_parser()
    args = parser.parse_args()
    log_file_path = os.path.join(args.working_dir, "ML.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    create_dir_if_not_exists(args.working_dir)
    main_data_dict = generate_data_dict_per_program(programs = ['fasttree','iqtree','raxml'], folder= args.main_data_folder, n_samp=args.n_main_samp)
    if args.use_val_data:
        val_data_dict = generate_data_dict_per_program(programs = ['fasttree','iqtree','raxml'], folder= args.validation_data_folder,
                                                       n_samp= args.n_val_samp)

    for ML_model in args.ML_model.split('_'):
        logging.info(f"Model = {ML_model}")
        model_working_dir = os.path.join(args.working_dir, ML_model)
        create_dir_if_not_exists(model_working_dir)
        for program in args.programs.split('_'):
            logging.info(f"Program = {program}")
            program_data = main_data_dict[program]
            logging.info(f"Number of trees in main data is {len(np.unique(program_data['tree_id']))}")
            validation_dict = {}
            if args.use_val_data:
                logging.info("Using validation data")
                program_validation_data = val_data_dict[program]

                program_validation_data = transform_data(program_validation_data,program)
                logging.info(f"Number of trees in validation is {len(np.unique(program_validation_data['tree_id']))}")
                for model_mode in np.unique(program_validation_data["model_mode"]):
                    validation_dict[f'val_{model_mode}'] = program_validation_data.loc[
                        program_validation_data.model_mode == model_mode].copy()
            working_dir = os.path.join(model_working_dir, program)
            create_dir_if_not_exists(working_dir)
            bootstrap_cols = get_bootstrap_col(program)

            sample_fracs = [float(frac) for frac in (args.sample_fracs).split('_')]
            all_model_merics = pd.DataFrame()
            if args.inc_sample_fracs and ML_model=='lightgbm':
                for sample_frac in sample_fracs:
                    logging.info(f"\n#Sample frac = {sample_frac}")
                    sample_frac_working_dir = os.path.join(working_dir, f"frac_{sample_frac}")
                    create_dir_if_not_exists(sample_frac_working_dir)




                    curr_model_metrics, groups_analysis = ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job,
                                                                      sample_frac_working_dir, sample_frac,
                                                                      compare_to_bootstrap_models=False,
                                                                      subsample_train=True, do_RFE=args.RFE,
                                                                      large_grid=False, name=f"frac_{sample_frac}",
                                                                      validation_dict=validation_dict, ML_model=ML_model)
                    all_model_merics = pd.concat([all_model_merics, curr_model_metrics])
            all_model_merics.to_csv(os.path.join(working_dir, 'all_models_performance.tsv'), sep=CSV_SEP)
            logging.info(f"Generating optimized final model")
            final_model_working_dir = os.path.join(working_dir, f"final_model")
            create_dir_if_not_exists(final_model_working_dir)
            final_models_performance,final_group_performance_full= ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job,
                                                               final_model_working_dir, sample_frac=-1,
                                                               subsample_train=False, do_RFE=args.RFE,
                                                               large_grid=args.full_grid, name=f"final_model",
                                                               validation_dict=validation_dict, ML_model=ML_model,
                                                               compare_to_bootstrap_models=True, extract_predictions=True)
            final_models_performance.to_csv(os.path.join(working_dir, 'final_model_performance.tsv'), sep=CSV_SEP)
            final_group_performance_full.to_csv(os.path.join(working_dir, 'groups_performance.tsv'), sep=CSV_SEP)

    #     print(test_metrics)


if __name__ == "__main__":
    main()
