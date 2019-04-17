# coding: utf-8

import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict


def load_data():
    df_all = pd.read_csv('./data/energydata_complete.csv')
    # Just Random variable for robustness that we drop
    df_all.drop(columns=['date', 'rv1', 'rv2'], inplace=True)
    return df_all


def get_train_test_data(df):
    target_column = "Appliances"
    # Split data
    train, test = train_test_split(df)

    train_x = train.drop([target_column], axis=1)
    test_x = test.drop([target_column], axis=1)
    train_y = train[target_column]
    test_y = test[target_column]

    return train_x, train_y, test_x, test_y


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def cv_eval_metrics(cv_model):
    rmse = np.sqrt(np.mean(cv_model['test_mse']))
    mae = np.mean(cv_model['test_mae'])
    r2 = np.mean(cv_model['test_r2'])
    return rmse, mae, r2


def scatter_plot_result(y_actual, y_pred, name):
    plt.scatter(y_actual, y_pred)
    plt.ylabel('Target predicted')
    plt.xlabel('True Target')
    plt.title(name)
    
    pos_x = y_actual.max() * 0.60
    pos_y = y_pred.max() * 0.90
    
    plt.text(pos_x, pos_y, r'$RMSE=%.2f, R^2$=%.2f, MAE=%.2f' % (np.sqrt(mean_squared_error(y_actual, y_pred)), 
                                              r2_score(y_actual, y_pred), 
                                              mean_absolute_error(y_actual, y_pred)))
    plt.plot([0, y_actual.max()], [0, y_actual.max()], ls="--", c=".3")
    plt.savefig('./scatter_results-{}.png'.format(name)) # Save to be included in Artefacts
    plt.close()


def plot_cv_results(iMetric, iResCV, model_name=None):
    # multiple line plot
    plt.plot(range(1,6)
             , iResCV["test_" + iMetric]
             , marker='o'
             , markerfacecolor='blue'
             , markersize=12, color='skyblue'
             , linewidth=4)
    plt.plot(range(1,6)
             , iResCV["train_" + iMetric]
             , marker='o'
             , markerfacecolor='red'
             , markersize=12
             , color='red'
             , linewidth=4)
    plt.xticks(range(1,6))
    plt.legend(('test','train'))
    plt.title(iMetric)
    if model_name is not None:
        plt.savefig('./cross_val_results-{}.png'.format(model_name))
    plt.close()
    
    
def log_metrics_classification(y_true, y_prediction):
    report = classification_report(y_true, y_prediction, output_dict=True)
    for class_ in ['0', '1']:
        for metric in report[class_]:
            log_name = class_ + '_' + metric
            # insert your code here 
            mlflow.log_metric(log_name, report[class_][metric])


def log_metrics_regression(cv_model):
    rmse, mae, r2 = cv_eval_metrics(cv_model)
    # log metrics here ~ 3 lines
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)


def set_mlfow_experiment(experiment_name):
    experiment_name = 'Default' if experiment_name is None else experiment_name
    mlflow.set_experiment(experiment_name)
    

def run_experiment(df, alpha, l1_ratio, experiment_name=None):
    # set exeperiment here ~ 1 line
    set_mlfow_experiment(experiment_name)
    
    # Split features and targets
    target_column = "Appliances"
    y = df[target_column]
    X = df.drop([target_column], axis=1)

    # Nb Cross Val
    CV = 5

    scorings = {
        'mse' : make_scorer(mean_squared_error),
        "mae" : make_scorer(mean_absolute_error),
        "r2" : make_scorer(r2_score)
    }

    with mlflow.start_run():
        print("Running with alpha: {} - l1_ratio: {}".format(alpha, l1_ratio))

        # fit models
        lr = ElasticNet(random_state=0, alpha=alpha, l1_ratio=l1_ratio)
        model = cross_validate(lr, X, y, scoring=scorings, cv=CV, n_jobs=-1, return_train_score=True)

        prediction_test = cross_val_predict(lr, X, y, cv=CV, n_jobs=-1, verbose=1)

        # log parameters
        # Your code here ~ 2 lines
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # log artifact
        scatter_name = 'ElasticNet'
        # save scatter plot as artifact here ~ 2 lines
        scatter_plot_result(y, prediction_test, scatter_name)
        mlflow.log_artifact('./scatter_results-{}.png'.format(scatter_name))
        
        # save cv_results ~ 1 line
        for metric in ['r2', 'mse', 'mae']:
            cv_result_name = metric + '_ElasticNet'
            # plot cv result here using cv_result_name to save the plot
            plot_cv_results(metric, model, model_name=cv_result_name)
            mlflow.log_artifact('./cross_val_results-{}.png'.format(cv_result_name))

        # log metrics
        log_metrics_regression(model)

        # log sklearn model
        train_x, train_y, test_x, test_y = get_train_test_data(df)
        lr.fit(train_x, train_y)
        # log the sklearn model here  ~ 1 line
        mlflow.sklearn.log_model(lr, 'elastic_net')


def test_range_params():
    df_all = load_data()
    for alpha in np.arange(0.1, 1, 0.2):
        for l1_ratio in np.arange(0.1, 1, 0.2):
            run_experiment(df_all, alpha, l1_ratio)


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    df_all = load_data()
    run_experiment(df_all, alpha, l1_ratio)


if __name__ == '__main__':
    main()
