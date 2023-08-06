from ML_utils.ML_algorithms_and_hueristics import *
import os




def main():
    full_data = pd.read_csv("/Users/noa/Workspace/edit_boot.tsv", sep = '\t')
    full_data["feature_ML_vs_pars"] = full_data["pars_bi_support_mean"]- full_data["ML_bi_support_mean"]
    #full_data["feature_pars_vs_b_pars"] = full_data["pars_support_mean"] - full_data["bootstrap_support_mean"]

    train, test = train_test_validation_splits(full_data, test_pct=0.3,
                                                         subsample_train=False, subsample_train_frac=-1)

    print(len(full_data['tree_id'].unique()))
    features = [col for col in full_data.columns if 'feature' in col or 'pars_support_mean' in col or 'pars_bi_support_mean' in col or 'ML_bi_support_mean' in col or 'ML_support_mean' in col ]+['partition_branch_vs_mean','partition_branch','partition_size','partition_size_ratio','partition_divergence','divergence_ratio']
    X_train = train[[col for col in train.columns if col in features]]
    X_test = test[[col for col in train.columns if col in features]]
    y_train = train["true_binary_support"]  # default_status
    y_test = test["true_binary_support"]

    model_path = os.path.join(os.getcwd(),'model')
    vi_path = os.path.join(os.getcwd(),'vi.tsv')
    model =lightgbm.LGBMClassifier(importance_type='gain').fit(X_train,y_train)

    var_impt = variable_importance(X_train, model)
    enrich_with_single_feature_metrics(var_impt, X_train, y_train, X_test, y_test)
    var_impt.to_csv(vi_path, sep=CSV_SEP)

    y_true = test["true_binary_support"]
    prob_predictions_test = model.predict_proba(X_test)[:, 1]
    predictions = model.predict((X_test))
    test_metrics = {   'AUC': roc_auc_score(y_true, prob_predictions_test),
         'logloss': log_loss(y_true, prob_predictions_test),
         'average_precision': average_precision_score(y_true, prob_predictions_test),
        'accuracy_score': accuracy_score(y_true, predictions),
        'precision': precision_score(y_true, predictions), 'recall': recall_score(y_true, predictions),
        'mcc': matthews_corrcoef(y_true, predictions)}
    print(test_metrics)





    for tree_id in test['tree_id'].unique():
        print(tree_id)
        tree_data_test = test[test.tree_id == tree_id]
        curr_y_true = tree_data_test["true_binary_support"]
        curr_X_test = tree_data_test[[col for col in train.columns if col in features]]
        prob_predictions_test = model.predict_proba(curr_X_test)[:, 1]
        predictions = model.predict(( curr_X_test))
        test_metrics = {#'AUC': roc_auc_score(curr_y_true, prob_predictions_test),
                    #'logloss': log_loss(curr_y_true, prob_predictions_test),
                    #'average_precision': average_precision_score(curr_y_true, prob_predictions_test),
                    'accuracy_score': accuracy_score(curr_y_true, predictions),
                    'precision': precision_score(curr_y_true, predictions), 'recall': recall_score(curr_y_true, predictions),
                    'mcc': matthews_corrcoef(curr_y_true, predictions)}
        #print(test_metrics)

if __name__ == "__main__":
    main()