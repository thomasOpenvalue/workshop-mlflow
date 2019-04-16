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
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


def scatter_plot_result(y_actual, y_pred, plot_name):
    plt.scatter(y_actual, y_pred)
    plt.ylabel('Target predicted')
    plt.xlabel('True Target')
    plt.title(plot_name)
    plt.text(500, 250, r'$RMSE=%.2f, R^2$=%.2f, MAE=%.2f' % (np.sqrt(mean_squared_error(y_actual, y_pred)),
                                                             r2_score(y_actual, y_pred),
                                                             mean_absolute_error(y_actual, y_pred)))
    plt.savefig(plot_name)
    plt.close()


def log_metrics_classification(y_true, y_prediction):
    report = classification_report(y_true, y_prediction, output_dict=True)
    for class_ in ['0', '1']:
        for metric in report[class_]:
            log_name = class_ + '_' + metric
            # insert your code here ~ 1 line


def log_metrics_regression(y_true, y_prediction):
    rmse, mae, r2 = eval_metrics(y_true, y_prediction)
    # log metrics here ~ 3 lines


def set_mlfow_experiment(experiment_name):
    experiment_name = 'Default' if experiment_name is None else experiment_name
    mlflow.set_experiment(experiment_name)    

    
def run_experiment(df, alpha, l1_ration experiment_name=None):
    # set exeperiment here ~ 1 line
    
    # Split data
    train_x, train_y, test_x, test_y = get_train_test_data(df)

    with mlflow.start_run():
        print("Running with alpha: {} - l1_ratio: {}".format(alpha, l1_ratio))

        # fit models
        lr = ElasticNet(random_state=0, alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(train_x, train_y)

        prediction_test = lr.predict(test_x)

        # log parameters
        # Your code here ~ 2 lines

        # log artifact
        scatter_name = './scatter_results-ElasticNet.png'
        # save scatter plot as artifact here ~ 2 lines

        # log metrics
        log_metrics_regression(test_y, prediction_test)

        # log sklearn model
        # log the sklearn model here  ~ 1 line


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
