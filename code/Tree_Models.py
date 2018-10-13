'''
This code is used to get Tree models result
1. Table A.1: Comparison of Predictive R2s for Machine Learning Algorithms in Simulations
  # Tree Models: RF\GBRT(H)
  # NN Models: NN1-5
  # PC: 50 & 100
  # Model: A & B
2. Table A.4: Comparison of Average Variable Importance in Simulations
  # Tree Models: RF\GBRT(H)
  # NN Models: NN1-5
  # PC: 50 & 100
  # Model: A & B
'''

import random as rn
import os
import numpy as np
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import PredefinedSplit


if __name__=="__main__":

    Path = 'F:\\MCS'
    Data50_fold = 'SimuData_p50'
    Data100_fold = 'SimuData_p100'

    x_file_list = []
    r1_file_list = []
    r2_file_list = []
    for i in range(1, 101, 1):
        x_file_list.append('c' + str(i))
        r1_file_list.append('r1_' + str(i))
        r2_file_list.append('r2_' + str(i))


    tableA1_all = np.zeros(shape=(3, 8))


    for foldnum, foldlist in enumerate([Data50_fold, Data100_fold]):

        tableA1_100 = np.zeros(shape=(3, 4))
        coef_100 = np.zeros(shape=(6, 100 * (foldnum + 1)))

        for x_num in range(100):
            print('This is round:', x_num)

            full_Path_x = Path + '\\MCSdata\\' + foldlist + '\\' + x_file_list[x_num]
            full_Path_y1 = Path + '\\MCSdata\\' + foldlist + '\\' + r1_file_list[x_num]
            full_Path_y2 = Path + '\\MCSdata\\' + foldlist + '\\' + r2_file_list[x_num]
            X_raw = pd.read_csv(full_Path_x + '.csv', header=None)
            Y1_raw = pd.read_csv(full_Path_y1 + '.csv', header=None)
            Y2_raw = pd.read_csv(full_Path_y2 + '.csv', header=None)
            X_array = np.array(X_raw)
            Y1_array = np.array(Y1_raw) * 100
            Y2_array = np.array(Y2_raw) * 100

            # define variables to store our result
            tableA1_temp = np.zeros(shape=(3, 4))
            coef_temp = np.zeros(shape=(6, 100 * (foldnum + 1)))


            # split the sample
            def split_data(df_X, df_Y):
                '''
                Data providing function
                Input:  df_X, X variable data, array like
                        df_Y, Y variable data, array like

                Return: X_traindata, All X train data include validation sample
                        Y_traindata, All Y train data include validation sample
                        X_vdata, All X validation data
                        Y_vdata, All Y validation data
                        X_testdata, All X test data
                        Y_testdata, All Y test data
                        mean_Ytrain, Mean value of Y train data exclude validation sample
                        X_traindata1, X train data exclude validation sample
                        Y_traindata1_demean, Demeaned Y train data exclude validation sample
                        Y_traindata_demean, Demeaned Y train data
                        Y_vdata_demean, Demeaned Y validation data
                        Y_testdata_demean Demeaned Y test data

                '''
                split_num = 200 * 60

                X_traindata1 = df_X[:split_num].copy()
                Y_traindata1 = df_Y[:split_num].copy()

                ##fitting sample
                X_traindata = df_X[:split_num * 2].copy()
                Y_traindata = df_Y[:split_num * 2].copy()

                X_vdata = df_X[split_num:split_num * 2].copy()
                Y_vdata = df_Y[split_num:split_num * 2].copy()
                X_testdata = df_X[split_num * 2:split_num * 3].copy()
                Y_testdata = df_Y[split_num * 2:split_num * 3].copy()

                for j in range(len(X_traindata[0])):
                    std_value = np.std(X_traindata1[:, j], ddof=1)
                    if std_value > 0:
                        X_traindata[:, j] = X_traindata[:, j] / std_value
                        X_traindata1[:, j] = X_traindata1[:, j] / std_value
                        X_vdata[:, j] = X_vdata[:, j] / std_value
                        X_testdata[:, j] = X_testdata[:, j] / std_value

                # Monthly Demean
                mean_Ytrain = np.mean(Y_traindata1)
                Y_traindata_demean = Y_traindata - mean_Ytrain
                Y_vdata_demean = Y_vdata - mean_Ytrain
                Y_testdata_demean = Y_testdata - mean_Ytrain
                Y_traindata1_demean = Y_traindata1 - mean_Ytrain

                return X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata, mean_Ytrain, \
                       X_traindata1, Y_traindata1, Y_traindata_demean, Y_vdata_demean, Y_testdata_demean, Y_traindata1_demean


            X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata, mean_Ytrain, X_traindata1, Y_traindata1, \
            Y_traindata_demean, Y_vdata_demean, Y_testdata_demean, Y_traindata1_demean = split_data(df_X=X_array,
                                                                                                    df_Y=Y1_array)
            ################################RF

            def rolling_model_RF(X_traindata=X_traindata,
                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                 X_traindata1=X_traindata1,
                                 Y_traindata1=np.ravel(Y_traindata1),
                                 X_testdata=X_testdata,
                                 Y_testdata=np.ravel(Y_testdata),
                                 mean_Ytrain=mean_Ytrain):


                # specify parameters and distributions to sample from

                split_num = 200 * 60
                num_valid_size = split_num
                test_fold = -1 * np.ones(len(X_traindata))
                test_fold[num_valid_size:] = 0
                ps = PredefinedSplit(test_fold)

                # specify parameters and distributions to sample from
                param_dist = {"max_features": sp_randint(5, 100),
                              "max_depth": sp_randint(3, 10),
                              "min_samples_split": sp_randint(10, 1000),
                              "min_samples_leaf": sp_randint(10, 1000),
                              "n_estimators": sp_randint(3, 100),
                              "oob_score": [True, False]
                              }

                clf_RF = RandomForestRegressor(random_state=100)

                # run randomized search
                n_iter_search = 50
                estim = RandomizedSearchCV(clf_RF, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring='r2',n_jobs=-1,
                                           cv=ps.split(), iid=False, random_state=100)

                estim.fit(X_traindata, Y_traindata_demean)
                best_estimator = estim.best_estimator_

                best_VIP = best_estimator.feature_importances_

                train_predict = best_estimator.predict(X_traindata1) + mean_Ytrain
                IS_score = r2_score(Y_traindata1, train_predict)

                test_predict = best_estimator.predict(X_testdata) + mean_Ytrain
                OOS_score = 1- np.sum((Y_testdata-test_predict)**2)/sum((Y_testdata-mean_Ytrain)**2)


                return IS_score, OOS_score, best_VIP


            IS_score, OOS_score, best_VIP = rolling_model_RF(X_traindata=X_traindata,
                                                             Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                             X_traindata1=X_traindata1,
                                                             Y_traindata1=np.ravel(Y_traindata1),
                                                             X_testdata=X_testdata,
                                                             Y_testdata=np.ravel(Y_testdata),
                                                             mean_Ytrain=mean_Ytrain)

            tableA1_temp[0, 0] = IS_score
            tableA1_temp[0, 1] = OOS_score
            print('RF Model a IS:', IS_score * 100)
            print('RF Model a OOS:', OOS_score * 100)

            coef_temp[0, :] = best_VIP


            ################################GBRT

            def rolling_model_GBRT(X_traindata=X_traindata,
                                   Y_traindata_demean=np.ravel(Y_traindata_demean),
                                   X_traindata1=X_traindata1,
                                   Y_traindata1=np.ravel(Y_traindata1),
                                   X_testdata=X_testdata,
                                   Y_testdata=np.ravel(Y_testdata),
                                   mean_Ytrain=mean_Ytrain,
                                   loss_type='ls'):
                # specify parameters and distributions to sample from

                split_num = 200 * 60
                num_valid_size = split_num
                test_fold = -1 * np.ones(len(X_traindata))
                test_fold[num_valid_size:] = 0
                ps = PredefinedSplit(test_fold)

                # specify parameters and distributions to sample from
                param_dist = {"max_features": sp_randint(5, 100),
                              "max_depth": sp_randint(3, 12),
                              "min_samples_split": sp_randint(10, 1000),
                              "min_samples_leaf": sp_randint(10, 1000),
                              "n_estimators": sp_randint(3, 100),
                              "learning_rate": uniform(0.001, 0.1),
                              "subsample": uniform(0.6, 0.4)
                              }

                clf_GBRT = GradientBoostingRegressor(loss=loss_type, random_state=100)

                # run randomized search
                n_iter_search = 50
                estim = RandomizedSearchCV(clf_GBRT, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring='r2', n_jobs=-1,
                                           cv=ps.split(), iid=False, random_state=100)

                estim.fit(X_traindata, Y_traindata_demean)
                best_estimator = estim.best_estimator_

                best_VIP = best_estimator.feature_importances_

                train_predict = best_estimator.predict(X_traindata1) + mean_Ytrain
                IS_score = r2_score(Y_traindata1, train_predict)
                test_predict = best_estimator.predict(X_testdata) + mean_Ytrain
                OOS_score = 1 - np.sum((Y_testdata - test_predict) ** 2) / sum((Y_testdata - mean_Ytrain) ** 2)

                return IS_score, OOS_score, best_VIP


            IS_score, OOS_score, best_VIP = rolling_model_GBRT(X_traindata=X_traindata,
                                                               Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                               X_traindata1=X_traindata1,
                                                               Y_traindata1=np.ravel(Y_traindata1),
                                                               X_testdata=X_testdata,
                                                               Y_testdata=np.ravel(Y_testdata),
                                                               mean_Ytrain=mean_Ytrain,
                                                               loss_type='ls')

            tableA1_temp[1, 0] = IS_score
            tableA1_temp[1, 1] = OOS_score
            print('GBRT Model a IS:', IS_score * 100)
            print('GBRT Model a OOS:', OOS_score * 100)

            coef_temp[1, :] = best_VIP

            ################################GBRT+H

            IS_score, OOS_score, best_VIP = rolling_model_GBRT(X_traindata=X_traindata,
                                                               Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                               X_traindata1=X_traindata1,
                                                               Y_traindata1=np.ravel(Y_traindata1),
                                                               X_testdata=X_testdata,
                                                               Y_testdata=np.ravel(Y_testdata),
                                                               mean_Ytrain=mean_Ytrain,
                                                               loss_type='huber')

            tableA1_temp[2, 0] = IS_score
            tableA1_temp[2, 1] = OOS_score
            print('GBRT+H Model a IS:', IS_score * 100)
            print('GBRT+H Model a OOS:', OOS_score * 100)

            coef_temp[2, :] = best_VIP

            ###############################################################################Model B
            X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata, mean_Ytrain, X_traindata1, Y_traindata1, \
            Y_traindata_demean, Y_vdata_demean, Y_testdata_demean, Y_traindata1_demean = split_data(df_X=X_array,
                                                                                                    df_Y=Y2_array)
            ################################Model B RF
            IS_score, OOS_score, best_VIP = rolling_model_RF(X_traindata=X_traindata,
                                                             Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                             X_traindata1=X_traindata1,
                                                             Y_traindata1=np.ravel(Y_traindata1),
                                                             X_testdata=X_testdata,
                                                             Y_testdata=np.ravel(Y_testdata),
                                                             mean_Ytrain=mean_Ytrain)

            tableA1_temp[0, 2] = IS_score
            tableA1_temp[0, 3] = OOS_score
            print('RF Model b IS:', IS_score * 100)
            print('RF Model b OOS:', OOS_score * 100)

            coef_temp[3, :] = best_VIP

            ################################Model B GBRT
            IS_score, OOS_score, best_VIP = rolling_model_GBRT(X_traindata=X_traindata,
                                                               Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                               X_traindata1=X_traindata1,
                                                               Y_traindata1=np.ravel(Y_traindata1),
                                                               X_testdata=X_testdata,
                                                               Y_testdata=np.ravel(Y_testdata),
                                                               mean_Ytrain=mean_Ytrain,
                                                               loss_type='ls')

            tableA1_temp[1, 2] = IS_score
            tableA1_temp[1, 3] = OOS_score
            print('GBRT Model b IS:', IS_score * 100)
            print('GBRT Model b OOS:', OOS_score * 100)

            coef_temp[4, :] = best_VIP


            ################################Model B GBRT+H
            IS_score, OOS_score, best_VIP = rolling_model_GBRT(X_traindata=X_traindata,
                                                               Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                               X_traindata1=X_traindata1,
                                                               Y_traindata1=np.ravel(Y_traindata1),
                                                               X_testdata=X_testdata,
                                                               Y_testdata=np.ravel(Y_testdata),
                                                               mean_Ytrain=mean_Ytrain,
                                                               loss_type='huber')

            tableA1_temp[2, 2] = IS_score
            tableA1_temp[2, 3] = OOS_score
            print('GBRT+H Model b IS:', IS_score * 100)
            print('GBRT+H Model b OOS:', OOS_score * 100)

            coef_temp[5, :] = best_VIP

            tableA1_100 = tableA1_100 + tableA1_temp
            coef_100 = coef_100 + coef_temp

        tableA1_all[:, 4 * foldnum:4 * (foldnum + 1)] = tableA1_100

        columns_list1 = ['RF', 'GBRT', 'GBRT+H','RF_B', 'GBRT_B', 'GBRT+H_B']
        tableA300 = pd.DataFrame(coef_100.T, columns=columns_list1)
        tableA300.to_csv(Path + '//output//' + foldlist + 'tableA3_Tree.csv')

    ################################define table
    columns_list = ['IS_a_50', 'OOS_a_50', 'IS_b_50', 'OOS_b_50',
                    'IS_a_100', 'OOS_a_100', 'IS_b_100', 'OOS_b_100']
    index_list = ['RF', 'GBRT', 'GBRT_H']

    tableA100 = pd.DataFrame(tableA1_all, columns=columns_list, index=index_list)
    tableA100.to_csv(Path + '//output//tableA1_Tree.csv')
