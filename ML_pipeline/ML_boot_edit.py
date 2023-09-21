from ML_utils.ML_algorithms_and_hueristics import *
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_MSA_seq_names
from sklearn.metrics import balanced_accuracy_score
import os
import matplotlib.pyplot as plt
import argparse



def get_test_metrics(y_true, predictions, prob_predictions):
    metrics = {
                    'balanced_accuracy_score': balanced_accuracy_score(y_true, predictions),
                    'accuracy_score': accuracy_score(y_true, predictions),
                    'precision': precision_score(y_true, predictions), 'recall': recall_score(y_true, predictions),
                    'mcc': matthews_corrcoef(y_true, predictions)}
    if prob_predictions is not None:
        prob_metrics = {'AUC': roc_auc_score(y_true, prob_predictions),
                        'logloss': log_loss(y_true, prob_predictions), 'average_precision': average_precision_score(y_true, prob_predictions), }
        metrics.update(prob_metrics)
    return metrics

def per_tree_analysis(test, features,model):
    balanced_accuracies = []
    for tree_id in test['tree_id'].unique():
        tree_data_test = test[test.tree_id == tree_id]
        curr_y_true = tree_data_test["true_binary_support"]
        curr_X_test = tree_data_test[[col for col in test.columns if col in features]]
        #prob_predictions = model.predict_proba(curr_X_test)[:, 1]
        predictions = model.predict(( curr_X_test))
        metrics = get_test_metrics(curr_y_true,predictions,None)
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


def overall_model_performance_analysis(working_dir,model, X_train, y_train, X_test, y_test, test, name):
    var_impt = variable_importance(X_train, model)
    enrich_with_single_feature_metrics(var_impt, X_train, y_train, X_test, y_test)
    vi_path = os.path.join(working_dir, f'{name}_vi.tsv')
    var_impt.to_csv(vi_path, sep=CSV_SEP)

    y_true = test["true_binary_support"]
    prob_predictions = model.predict_proba(X_test)[:, 1]
    predictions = model.predict((X_test))
    test_metrics = get_test_metrics(y_true, predictions, prob_predictions)
    return test_metrics


def raw_bootstrap_performance(test, bootstrap_col, y_true):
    prob_predictions_test_boot = test[bootstrap_col] / 100
    predictions_test_boot = test[bootstrap_col] > 0.5
    test_metrics_boot = get_test_metrics(y_true, predictions_test_boot, prob_predictions_test_boot)
    return test_metrics_boot

def bootstrap_feature_analysis(working_dir,bootstrap_col,train,test, y_train, y_test, features):
    res = pd.DataFrame()
    X_train_only_boot = train[[bootstrap_col]]
    X_test_inc_boot = test[[col for col in train.columns if col in features] + [bootstrap_col]]
    X_train_inc_boot = train[[col for col in train.columns if col in features] + [bootstrap_col]]
    X_test_only_boot = test[[bootstrap_col]]
    model_inc_boot = model_pipeline(working_dir,X_train_inc_boot, y_train, name=f"inc_boot_{bootstrap_col}")
    mp_inc_boot = overall_model_performance_analysis(working_dir,model_inc_boot, X_train_inc_boot, y_train, X_test_inc_boot,
                                                     y_test, test, name=f"inc_boot_{bootstrap_col}")
    mp_inc_boot["type"] = f'{bootstrap_col} including bootstrap'
    res = res.append(mp_inc_boot, ignore_index= True)
    print(f"Model performance including boot: \n {mp_inc_boot}")
    raw_boot_performance = raw_bootstrap_performance(test, bootstrap_col, y_test)
    raw_boot_performance["type"] = f'{bootstrap_col} using raw bootstrap values'
    res = res.append(raw_boot_performance, ignore_index=True)
    print(f"Raw model performance is: \n {raw_boot_performance}")
    model_only_boot = model_pipeline(working_dir,X_train_only_boot, y_train, name=f"only_boot_{bootstrap_col}")
    mp_only_boot = overall_model_performance_analysis(working_dir,model_only_boot, X_train_only_boot, y_train, X_test_only_boot,
                                                      y_test,
                                                      test, name=f"only_boot_{bootstrap_col}")
    mp_only_boot["type"] = f'{bootstrap_col} using single bootstrap feature'
    res = res.append(mp_only_boot, ignore_index=True)
    print(f"Model performance using only boot: \n{mp_only_boot}")
    return res


def get_bootstrap_col(program):
    if program == 'raxml':
        # full_data["feature_ML_vs_pars"] = full_data["feature_parsimony_trees_binary_mean"]- full_data["feature_all_ML_ete_binary_mean"]
        bootstrap_cols = ['bootstrap_support']
    elif program == 'iqtree':
        bootstrap_cols = ['bootstrap_support', 'feature_aLRT_iqtree_support', 'feature_aBayes_iqtree_support']
    elif program == 'fasttree':
        bootstrap_cols = ['feature_standard_fasttree_boot_support', 'bootstrap_support']
    return bootstrap_cols




def ML_pipeline(program_data, bootstrap_cols, cpus_per_main_job, working_dir, sample_frac,subsample_train, do_RFE , large_grid, name, validation_data = None):
    program_data = program_data.dropna(axis=1, how='all')
    program_data = program_data.dropna(axis=0)
    train, test = train_test_validation_splits(program_data, test_pct=0.3,
                                               subsample_train=subsample_train, subsample_train_frac=sample_frac)

    print(len(program_data['tree_id'].unique()))
    features = [col for col in program_data.columns if
                'feature' in col and col not in bootstrap_cols]  # +['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']

    X_train = train[[col for col in train.columns if col in features]]
    X_test = test[[col for col in train.columns if col in features]]
    y_train = train["true_binary_support"]  # default_status
    y_test = test["true_binary_support"]

    groups = train["tree_id"]
    model = ML_model(X_train, groups, y_train, n_jobs=cpus_per_main_job, path=os.path.join(working_dir, f'model_{sample_frac}'),
                     classifier=True, model='lightgbm', calibrate=True, name=name, large_grid=large_grid, do_RFE=do_RFE,
                     n_cv_folds=3)
    val_expended_dict = {}
    if validation_data is not None:
        for tree_ind in validation_data["tree_id"]:
            tree_validation_data = validation_data.loc[validation_data.tree_id==tree_ind]
            X_val = tree_validation_data[[col for col in tree_validation_data.columns if col in features]]
            y_val = tree_validation_data["true_binary_support"]
            val_expended_dict[tree_ind] = {'X_val': X_val, 'y_val': y_val}
    else:
        X_val = None
        y_val = None

    print_model_statistics_pipeline(model, X_train, X_test, y_train, y_test, X_val, y_val,  val_expanded_dict=val_expended_dict,
                                    is_classification=True,
                                    vi_path=os.path.join(working_dir, 'vi.tsv'),
                                    error_vs_size_path=os.path.join(working_dir, 'erorr_vs_size.tsv'),
                                    classification_metrics_path=os.path.join(working_dir,
                                                                             'all_clasissification_metrics.tsv'),
                                    validation_metrics_path=os.path.join(working_dir,
                                                                             'validation_metrics.tsv'),
                                    group_metrics_path=os.path.join(working_dir, 'group_metrics.tsv'), name="testing",
                                    sampling_frac=sample_frac,
                                    feature_importance=True)


def transform_data(df):
    df['true_binary_support'] = df['true_support'] == 1
    df['feature_msa_n_seq'] = df['feature_n_unique_seq']
    df.drop(columns = ['feature_n_unique_seq'], inplace= True)
    #+[col for col in df.columns if 'msa_entropy' in col]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type = str, default = os.getcwd())
    parser.add_argument('--cpus_per_main_job', type = int, default=1)
    parser.add_argument('--sample_fracs', type = str, default='0.3_0.7_1')

    full_data = pd.read_csv("/Users/noa/Workspace/bootstrap_results/job_raw_data_with_features.tsv", sep = '\t')

    validation_raxml = pd.read_csv("/Users/noa/Workspace/bootstrap_results/simulations_df_raxml.tsv", sep = '\t')
    transform_data(validation_raxml)
    validation_iqtree = pd.read_csv("/Users/noa/Workspace/bootstrap_results/simulations_df_iqtree.tsv", sep='\t')
    transform_data(validation_iqtree)
    validation_fasttree = pd.read_csv("/Users/noa/Workspace/bootstrap_results/simulations_df_fasttree.tsv", sep='\t')
    transform_data(validation_fasttree)
    validation_dict = {'raxml': validation_raxml, 'iqtree': validation_iqtree, 'fasttree': validation_fasttree}

    transform_data(full_data)
    args = parser.parse_args()
    for program in full_data['program'].unique():
        program_data = full_data.loc[full_data.program == program]
        working_dir = os.path.join(args.working_dir, program)
        create_dir_if_not_exists(working_dir)
        bootstrap_cols = get_bootstrap_col(program)
        program_data = program_data.dropna(axis=1, how='all')
        program_data = program_data.dropna(axis=0)
        sample_fracs = [float(frac) for frac in (args.sample_fracs).split('_')]
        for sample_frac in sample_fracs:
            ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job, working_dir, sample_frac,subsample_train = True,do_RFE = False, large_grid = False, name = f"frac_{sample_frac}")

        ML_pipeline(program_data, bootstrap_cols, args.cpus_per_main_job, working_dir, sample_frac = -1,subsample_train = False, do_RFE=False,
                    large_grid=False, name=f"final_model", validation_data = validation_dict[program])




'''

        program_results = pd.DataFrame()
        model_performance = overall_model_performance_analysis(working_dir,model["best_model"], X_train, y_train, X_test, y_test, test,name = "standard")
        model_performance['type'] = 'features_based_analysis'
        print(f"Overall model performance: \n { model_performance}")
        program_results = program_results.append(model_performance, ignore_index= True)
        for bootstrap_col in bootstrap_cols:
            print((f"\n\n####{bootstrap_col}:"))
            boot_performance = bootstrap_feature_analysis(working_dir,bootstrap_col, train, test, y_train, y_test, features)
            program_results = pd.concat([program_results, boot_performance])
        program_results.to_csv(f"{working_dir}_performance_res.csv")


    #per_tree_analysis(test, features, model)

'''




    #     print(test_metrics)

if __name__ == "__main__":
    main()