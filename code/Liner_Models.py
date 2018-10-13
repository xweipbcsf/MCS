'''
This code is used to get liner models result
1. Table A.1: Comparison of Predictive R2s for Machine Learning Algorithms in Simulations
  # Liner Models: OLS(H)\PCR\PLS\Lasso(H)\Ridge(H)\ENet(H)\Oracle
  # PC: 50 & 100
  # Model: A & B
2. Table A.3: Comparison of Average Variable Selection Frequencies in Simulations
  # Liner Models: Lasso(H)\ENet(H)
  # PC: 50 & 100
  # Model: A & B
'''

# 0 set the environment
import random as rn
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

from sklearn.metrics import r2_score

if __name__=="__main__":

    np.random.seed(42)
    rn.seed(123)
    Path = 'F:\\MCS'


    # define loop info

    Data50_fold = 'SimuData_p50'
    Data100_fold = 'SimuData_p100'

    x_file_list = []
    r1_file_list = []
    r2_file_list = []
    for i in range(1, 101, 1):
        x_file_list.append('c' + str(i))
        r1_file_list.append('r1_' + str(i))
        r2_file_list.append('r2_' + str(i))

    tableA1_all = np.zeros(shape=(11, 8))


    for foldnum, foldlist in enumerate([Data50_fold, Data100_fold]):

        tableA1_100 = np.zeros(shape=(11, 4))
        coef_100 = np.zeros(shape=(8, 100 * (foldnum + 1)))

        for x_num in range(100):
            print('This is round:', x_num)

            full_Path_x = Path + '\\MCSdata\\' + foldlist + '\\' + x_file_list[x_num]
            full_Path_y1 = Path + '\\MCSdata\\' + foldlist + '\\' + r1_file_list[x_num]
            full_Path_y2 = Path + '\\MCSdata\\' + foldlist + '\\' + r2_file_list[x_num]
            X_raw = pd.read_csv(full_Path_x + '.csv', header=None)
            Y1_raw = pd.read_csv(full_Path_y1 + '.csv', header=None)
            Y2_raw = pd.read_csv(full_Path_y2 + '.csv', header=None)
            X_array = np.array(X_raw)
            Y1_array = np.array(Y1_raw) *100
            Y2_array = np.array(Y2_raw) *100

            # define variables to store our result
            tableA1_temp = np.zeros(shape=(11, 4))
            coef_temp = np.zeros(shape=(8, 100 * (foldnum + 1)))


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


            ################################Oracle Model

            def rolling_model_Oracle(X_traindata1=X_traindata1,
                                     Y_traindata1=Y_traindata1,
  #                                   Y_traindata1_demean=Y_traindata1_demean,
                                     X_testdata=X_testdata,
                                     Y_testdata=Y_testdata,
                                     mean_Ytrain=mean_Ytrain,
                                     model='A',
                                     foldnum=foldnum):
                '''
                Oracle Model function:
                    fit the model with X_traindata1 & Y_traindata1_demean;
                    predict the model with X_traindata1 & X_testdata;
                    calculate the IS and OOS score with prediction and Y_traindata1 & Y_testdata
                Input:
                    :param X_traindata1: same with split_data function
                    :param Y_traindata1: same with split_data function
                    :param Y_traindata1_demean: same with split_data function
                    :param X_testdata: same with split_data function
                    :param Y_testdata: same with split_data function
                    :param mean_Ytrain: same with split_data function
                    :param model: A or B
                    :param foldnum: 0 or 1
                Return:
                    IS_score: in-sample R square
                    OOS_score: out-of-sample R square
                '''
                theta_w = 0.02
                X_traindata1_Oracle = np.zeros(shape=(200 * 60, 3))
                X_testdata_Oracle = np.zeros(shape=(200 * 60, 3))
                if model == 'A':
                    X_traindata1_Oracle[:, 0] = theta_w * X_traindata1[:, 0]
                    X_traindata1_Oracle[:, 1] = theta_w * X_traindata1[:, 1]
                    X_traindata1_Oracle[:, 2] = theta_w * X_traindata1[:, (foldnum + 1) * 50 + 2]
                    X_testdata_Oracle[:, 0] = theta_w * X_testdata[:, 0]
                    X_testdata_Oracle[:, 1] = theta_w * X_testdata[:, 1]
                    X_testdata_Oracle[:, 2] = theta_w * X_testdata[:, (foldnum + 1) * 50 + 2]
                else:
                    X_traindata1_Oracle[:, 0] = 2 * theta_w * (X_traindata1[:, 0] ** 2)
                    X_traindata1_Oracle[:, 1] = 1.5 * theta_w * (X_traindata1[:, 0] * X_traindata1[:, 1])
                    X_traindata1_Oracle[:, 2] = 0.6 * theta_w * np.sign(X_traindata1[:, (foldnum + 1) * 50 + 2])
                    X_testdata_Oracle[:, 0] = 2 * theta_w * (X_testdata[:, 0] ** 2)
                    X_testdata_Oracle[:, 1] = 1.5 * theta_w * (X_testdata[:, 0] * X_testdata[:, 1])
                    X_testdata_Oracle[:, 2] = 0.6 * theta_w * np.sign(X_testdata[:, (foldnum + 1) * 50 + 2])

                clf_OLS = LinearRegression(n_jobs=2, fit_intercept=False)
                # Orcal One
                # clf_OLS.fit(X_traindata1_Oracle, Y_traindata1_demean)
                # train_predict = clf_OLS.predict(X_traindata1_Oracle) + mean_Ytrain
                # IS_score = r2_score(Y_traindata1, train_predict)
                # test_predict = clf_OLS.predict(X_testdata_Oracle) + mean_Ytrain
                # OOS_score = 1 - np.sum((Y_testdata - test_predict) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

                #Orcal Paper
                clf_OLS.fit(X_traindata1_Oracle, Y_traindata1)
                train_predict = clf_OLS.predict(X_traindata1_Oracle)
                IS_score = r2_score(Y_traindata1, train_predict)
                test_predict = clf_OLS.predict(X_testdata_Oracle)
                OOS_score = 1 - np.sum((Y_testdata - test_predict) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

                return IS_score, OOS_score


            IS_score, OOS_score = rolling_model_Oracle()

            tableA1_temp[0, 0] = IS_score
            tableA1_temp[0, 1] = OOS_score
            print('Oracle Model a IS:', IS_score * 100)
            print('Oracle Model a OOS:', OOS_score * 100)


            ################################OLS Model

            def rolling_model_OLS(X_traindata1=X_traindata1,
                                  Y_traindata1=np.ravel(Y_traindata1),
                                  Y_traindata1_demean=np.ravel(Y_traindata1_demean),
                                  X_testdata=X_testdata,
                                  Y_testdata=np.ravel(Y_testdata),
                                  mean_Ytrain=mean_Ytrain):
                '''
                OLS Model function:
                    fit the model with X_traindata1 & Y_traindata1_demean;
                    predict the model with X_traindata1 & X_testdata;
                    calculate the IS and OOS score with prediction and Y_traindata1 & Y_testdata
                Input:
                    :param X_traindata1: same with split_data function
                    :param Y_traindata1: same with split_data function
                    :param Y_traindata1_demean: same with split_data function
                    :param X_testdata: same with split_data function
                    :param Y_testdata: same with split_data function
                    :param mean_Ytrain: same with split_data function
                Return:
                    IS_score: in-sample R square
                    OOS_score: out-of-sample R square
                '''
                clf_OLS = LinearRegression(n_jobs=2, fit_intercept=False)
                clf_OLS.fit(X_traindata1, Y_traindata1_demean)

                train_predict = clf_OLS.predict(X_traindata1) + mean_Ytrain
                IS_score = r2_score(Y_traindata1, train_predict)

                test_predict = clf_OLS.predict(X_testdata) + mean_Ytrain
                OOS_score = 1 - np.sum((Y_testdata - test_predict) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

                return IS_score, OOS_score


            IS_score, OOS_score = rolling_model_OLS()

            tableA1_temp[1, 0] = IS_score
            tableA1_temp[1, 1] = OOS_score
            print('OLS Model a IS:', IS_score * 100)
            print('OLS Model a OOS:', OOS_score * 100)


            ################################define Liner Models

            def rolling_model_Liner(X_traindata=X_traindata,
                                    Y_traindata_demean=np.ravel(Y_traindata_demean),
                                    X_traindata1=X_traindata1,
                                    Y_traindata1=np.ravel(Y_traindata1),
                                    X_testdata=X_testdata,
                                    Y_testdata=np.ravel(Y_testdata),
                                    mean_Ytrain=mean_Ytrain,
                                    loss_type='huber',
                                    penalty_type='none'):
                '''
                Liner Model function:
                    This model function contains 7 liner models:OLSH\Lasso(H)\Ridge(H)\ENet(H)
                :param X_traindata: same with split_data function
                :param Y_traindata_demean: same with split_data function
                :param X_traindata1: same with split_data function
                :param Y_traindata1: same with split_data function
                :param X_testdata: same with split_data function
                :param Y_testdata: same with split_data function
                :param mean_Ytrain: same with split_data function
                :param loss_type:
                        'squared_loss': stands for mean squared loss function
                        'huber': stands for use Huber loss function
                :param penalty_type:
                        'l1': stands for lasso
                        'l2': stands for ridge
                        'elasticnet': stands for ENet
                :return:
                        IS_score: in-sample R square
                        OOS_score: out-of-sample R square
                        best_coef: correlation coefficent of each variable ,array
                '''

                split_num = 200 * 60
                num_valid_size = split_num
                test_fold = -1 * np.ones(len(X_traindata))
                test_fold[num_valid_size:] = 0
                ps = PredefinedSplit(test_fold)


                if penalty_type == 'l2':
                    '''
                    eta0 = 0.001,  # The initial learning rate[default 0.0]
                    # huber loss parameter
                    epsilon = 20,  # determines the threshold at which it becomes less important to get the prediction exactly right
                    alpha = 2,  # Constant that multiplies the regularization term
                    # Elastic Netparameter
                    l1_ratio = 0,  # l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15

                    '''

                    param_dist = {'alpha': sp_randint(1, 5),
                                  'l1_ratio': uniform(0.1, 0.5),
                                  'eta0': uniform(0.001, 0.001),
                                  'epsilon': sp_randint(10, 20),
                                  }


                else:

                    '''
                    # huber loss parameter
                    eta0 = 0.001
                    epsilon = 20,  # determines the threshold at which it becomes less important to get the prediction exactly right
                    alpha = 0.21,  # Constant that multiplies the regularization term
                    # Elastic Netparameter
                    l1_ratio = 1,  # l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15
                    '''

                    param_dist = {'alpha': uniform(0.1, 0.4),
                                  'l1_ratio': uniform(0.6, 0.4),
                                  'eta0': uniform(0.001, 0.001),
                                  'epsilon': sp_randint(10, 30),
                                  }

                clf = SGDRegressor(
                    loss=loss_type,
                    penalty=penalty_type,
                    learning_rate='adaptive',  # eta = eta0, as long as the training keeps decreasing
                    n_iter_no_change=5,  # Number of iterations with no improvement to wait before early stopping.
                    early_stopping=True,
                    # Whether to use early stopping to terminate training when validation score is not improving
                    validation_fraction=0.5,
                    # The proportion of training data to set aside as validation set for early stopping
                    fit_intercept = False,  # Whether the intercept should be estimated or not
                    max_iter = 5000,
                    # The maximum number of passes over the training data (aka epochs) Defaults to 5. Defaults to 1000 from 0.21, or if tol is not None.
                    tol = 0.0001,
                    # The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). Defaults to None. Defaults to 1e-3 from 0.21.
                    shuffle = False,
                    random_state = 100)

                # run randomized search
                n_iter_search = 30
                estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring='r2',
                                           cv=ps.split(), iid=False,
                                           random_state=100, n_jobs=-1)

                estim.fit(X_traindata, Y_traindata_demean)
                best_estimator = estim.best_estimator_
                best_coef = best_estimator.coef_

                train_predict = best_estimator.predict(X_traindata1) + mean_Ytrain
                IS_score = r2_score(Y_traindata1, train_predict)
                test_predict = best_estimator.predict(X_testdata) + mean_Ytrain
                OOS_score = 1 - np.sum((Y_testdata - test_predict) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

                return IS_score, OOS_score, best_coef


            ################################OLS+H
            IS_score, OOS_score, best_coef = rolling_model_Liner()

            tableA1_temp[2, 0] = IS_score
            tableA1_temp[2, 1] = OOS_score
            print('OLS+H Model a IS:', IS_score * 100)
            print('OLS+H Model a OOS:', OOS_score * 100)

            ################################Lasso
            IS_score, OOS_score, best_coef = rolling_model_Liner(loss_type='squared_loss', penalty_type='l1')

            tableA1_temp[5, 0] = IS_score
            tableA1_temp[5, 1] = OOS_score
            print('Lasso Model a IS:', IS_score * 100)
            print('Lasso Model a OOS:', OOS_score * 100)


            def change_dummy(coef_temp):
                for h in range(len(coef_temp)):
                    if coef_temp[h] != 0:
                        coef_temp[h] = int(1)


            change_dummy(best_coef)
            coef_temp[0, :] = best_coef

            ################################Lasso+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(loss_type='huber', penalty_type='l1')

            tableA1_temp[6, 0] = IS_score
            tableA1_temp[6, 1] = OOS_score
            print('Lasso+H Model a IS:', IS_score * 100)
            print('Lasso+H Model a OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[1, :] = best_coef

            ################################Ridge
            IS_score, OOS_score, best_coef = rolling_model_Liner(loss_type='squared_loss', penalty_type='l2')

            tableA1_temp[7, 0] = IS_score
            tableA1_temp[7, 1] = OOS_score
            print('Ridge Model a IS:', IS_score * 100)
            print('Ridge Model a OOS:', OOS_score * 100)

            ################################Ridge+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(loss_type='huber', penalty_type='l2')

            tableA1_temp[8, 0] = IS_score
            tableA1_temp[8, 1] = OOS_score
            print('Ridge+H Model a IS:', IS_score * 100)
            print('Ridge+H Model a OOS:', OOS_score * 100)

            ################################ENet
            IS_score, OOS_score, best_coef = rolling_model_Liner(loss_type='squared_loss', penalty_type='elasticnet')

            tableA1_temp[9, 0] = IS_score
            tableA1_temp[9, 1] = OOS_score
            print('ENet Model a IS:', IS_score * 100)
            print('ENet Model a OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[2, :] = best_coef

            ################################ENet+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(loss_type='huber', penalty_type='elasticnet')

            tableA1_temp[10, 0] = IS_score
            tableA1_temp[10, 1] = OOS_score
            print('ENet+H Model a IS:', IS_score * 100)
            print('ENet+H Model a OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[3, :] = best_coef

            #
            ################################define PCR Model
            def rolling_model_PCR(X_traindata=X_traindata,
                                  Y_traindata_demean=np.ravel(Y_traindata_demean),
                                  Y_traindata=np.ravel(Y_traindata),
                                  X_testdata=X_testdata,
                                  Y_testdata=np.ravel(Y_testdata),
                                  mean_Ytrain=mean_Ytrain):
                '''
                PCR Model function: same with split_data function
                :param X_traindata: same with split_data function
                :param Y_traindata_demean: same with split_data function
                :param Y_traindata: same with split_data function
                :param X_testdata: same with split_data function
                :param Y_testdata: same with split_data function
                :param mean_Ytrain: same with split_data function
                :return:
                        IS_score: in-sample R square
                        OOS_score: out-of-sample R square
                '''

                split_num = 200 * 60
                v_score = pd.DataFrame(index=range(1, 32, 1), columns=['Is_score', 'v_score', 'test_score'])

                for j in range(1, 31, 1):
                    pca = PCA(n_components=j)
                    X_reduced_train = pca.fit_transform(X_traindata)
                    X_reduced_test = pca.transform(X_testdata)

                    X_reduced_train = X_reduced_train[:, :j].copy()
                    X_reduced_test = X_reduced_test[:, :j].copy()

                    clf_OLS = LinearRegression(n_jobs=2)
                    clf_OLS.fit(X_reduced_train, Y_traindata_demean)
                    # refit best model & fit

                    train_pred = clf_OLS.predict(X_reduced_train[:split_num, :]) + mean_Ytrain
                    Is_performance_score = r2_score(Y_traindata[:split_num], train_pred)
                    v_score.iloc[j, 0] = Is_performance_score

                    v_pred = clf_OLS.predict(X_reduced_train[split_num:split_num * 2]) + mean_Ytrain
                    v_performance_score = 1 - np.sum((Y_traindata[split_num:split_num * 2] - v_pred) ** 2) / np.sum((Y_traindata[split_num:split_num * 2] - mean_Ytrain) ** 2)
                    v_score.iloc[j, 1] = v_performance_score

                    test_pre_y_array = clf_OLS.predict(X_reduced_test) + mean_Ytrain
                    test_performance_score = r2_score(Y_testdata, test_pre_y_array)
                    v_score.iloc[j, 2] = test_performance_score

                v_score.dropna(inplace=True)
                v_score.sort_values(by=['v_score'], inplace=True, ascending=False)
                IS_score = v_score.iloc[0, 0]
                OOS_score = v_score.iloc[0, 2]
                return IS_score, OOS_score


            ################################PCR
            IS_score, OOS_score = rolling_model_PCR()
            print('PCR Model a IS:', IS_score * 100)
            print('PCR Model a OOS:', OOS_score * 100)
            tableA1_temp[3, 0] = IS_score
            tableA1_temp[3, 1] = OOS_score


            ################################define PLS Model

            def rolling_model_PLS(X_traindata=X_traindata,
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
                param_dist = {'n_components': sp_randint(1, 31),
                              'max_iter': sp_randint(50, len(X_traindata)),
                              'tol': [0.0001, 0.00001, 0.000001, 0.0000001]}

                PLS_model = PLSRegression(scale=False)

                # run gridsearchcv make_scorer(r2_score)
                n_iter_search = 50
                estim = RandomizedSearchCV(PLS_model, param_distributions=param_dist, scoring='r2',
                                           cv=ps.split(), iid=False, n_jobs=-1, n_iter=n_iter_search)

                estim.fit(X_traindata, Y_traindata_demean)
                best_estimator = estim.best_estimator_

                train_predict = best_estimator.predict(X_traindata1) + mean_Ytrain
                IS_score = r2_score(Y_traindata1, train_predict)

                test_predict = best_estimator.predict(X_testdata) + mean_Ytrain
                test_predict = test_predict[:,0]
                OOS_score = 1 - np.sum((Y_testdata - test_predict) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

                return IS_score, OOS_score


            ################################PLS
            IS_score, OOS_score = rolling_model_PLS()
            print('PLS Model a IS:', IS_score * 100)
            print('PLS Model a OOS:', OOS_score * 100)
            tableA1_temp[4, 0] = IS_score
            tableA1_temp[4, 1] = OOS_score

            ##################################################################Model B

            X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata, mean_Ytrain, X_traindata1, Y_traindata1, \
            Y_traindata_demean, Y_vdata_demean, Y_testdata_demean, Y_traindata1_demean = split_data(df_X=X_array,
                                                                                                    df_Y=Y2_array)

            ################################Oracle
            IS_score, OOS_score = rolling_model_Oracle(X_traindata1=X_traindata1,
                                                       Y_traindata1=Y_traindata1,
                                                       X_testdata=X_testdata,
                                                       Y_testdata=Y_testdata,
                                                       mean_Ytrain=mean_Ytrain, model='B', foldnum=foldnum)

            tableA1_temp[0, 2] = IS_score
            tableA1_temp[0, 3] = OOS_score
            print('Oracle Model b IS:', IS_score * 100)
            print('Oracle Model b OOS:', OOS_score * 100)

            ################################OLS
            IS_score, OOS_score = rolling_model_OLS(X_traindata1=X_traindata1,
                                                    Y_traindata1=np.ravel(Y_traindata1),
                                                    Y_traindata1_demean=np.ravel(Y_traindata1_demean),
                                                    X_testdata=X_testdata,
                                                    Y_testdata=np.ravel(Y_testdata),
                                                    mean_Ytrain=mean_Ytrain)

            tableA1_temp[1, 2] = IS_score
            tableA1_temp[1, 3] = OOS_score
            print('OLS Model b IS:', IS_score * 100)
            print('OLS Model b OOS:', OOS_score * 100)

            ################################OLS+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='huber',
                                                                 penalty_type='none')

            tableA1_temp[2, 2] = IS_score
            tableA1_temp[2, 3] = OOS_score
            print('OLS+H Model b IS:', IS_score * 100)
            print('OLS+H Model b OOS:', OOS_score * 100)

            ################################PCR
            IS_score, OOS_score = rolling_model_PCR(X_traindata=X_traindata,
                                                    Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                    Y_traindata=np.ravel(Y_traindata),
                                                    X_testdata=X_testdata,
                                                    Y_testdata=np.ravel(Y_testdata),
                                                    mean_Ytrain=mean_Ytrain)
            print('PCR Model b IS:', IS_score * 100)
            print('PCR Model b OOS:', OOS_score * 100)
            tableA1_temp[3, 2] = IS_score
            tableA1_temp[3, 3] = OOS_score

            ################################PLS
            IS_score, OOS_score = rolling_model_PLS(X_traindata=X_traindata,
                                                    Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                    X_traindata1=X_traindata1,
                                                    Y_traindata1=np.ravel(Y_traindata1),
                                                    X_testdata=X_testdata,
                                                    Y_testdata=np.ravel(Y_testdata),
                                                    mean_Ytrain=mean_Ytrain)
            print('PLS Model b IS:', IS_score * 100)
            print('PLS Model b OOS:', OOS_score * 100)
            tableA1_temp[4, 2] = IS_score
            tableA1_temp[4, 3] = OOS_score

            ################################Lasso
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='squared_loss',
                                                                 penalty_type='l1')

            tableA1_temp[5, 2] = IS_score
            tableA1_temp[5, 3] = OOS_score
            print('Lasso Model b IS:', IS_score * 100)
            print('Lasso Model b OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[4, :] = best_coef

            ################################Lasso+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='huber',
                                                                 penalty_type='l1')

            tableA1_temp[6, 2] = IS_score
            tableA1_temp[6, 3] = OOS_score
            print('Lasso+H Model b IS:', IS_score * 100)
            print('Lasso+H Model b OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[5, :] = best_coef

            ################################Ridge
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='squared_loss',
                                                                 penalty_type='l2')

            tableA1_temp[7, 2] = IS_score
            tableA1_temp[7, 3] = OOS_score
            print('Ridge Model b IS:', IS_score * 100)
            print('Ridge Model b OOS:', OOS_score * 100)

            ################################Ridge+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='huber',
                                                                 penalty_type='l2')

            tableA1_temp[8, 2] = IS_score
            tableA1_temp[8, 3] = OOS_score
            print('Ridge+H Model b IS:', IS_score * 100)
            print('Ridge+H Model b OOS:', OOS_score * 100)

            ################################ENet
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='squared_loss',
                                                                 penalty_type='elasticnet')

            tableA1_temp[9, 2] = IS_score
            tableA1_temp[9, 3] = OOS_score
            print('ENet Model b IS:', IS_score * 100)
            print('ENet Model b OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[6, :] = best_coef

            ################################ENet+H
            IS_score, OOS_score, best_coef = rolling_model_Liner(X_traindata=X_traindata,
                                                                 Y_traindata_demean=np.ravel(Y_traindata_demean),
                                                                 X_traindata1=X_traindata1,
                                                                 Y_traindata1=np.ravel(Y_traindata1),
                                                                 X_testdata=X_testdata,
                                                                 Y_testdata=np.ravel(Y_testdata),
                                                                 mean_Ytrain=mean_Ytrain, loss_type='huber',
                                                                 penalty_type='elasticnet')

            tableA1_temp[10, 2] = IS_score
            tableA1_temp[10, 3] = OOS_score
            print('ENet+H Model b IS:', IS_score * 100)
            print('ENet+H Model b OOS:', OOS_score * 100)

            change_dummy(best_coef)
            coef_temp[7, :] = best_coef
            tableA1_100 = tableA1_100 + tableA1_temp
            coef_100 = coef_100 + coef_temp

        tableA1_all[:, 4 * foldnum:4 * (foldnum + 1)] = tableA1_100

        columns_list1 = ['Lasso', 'Lasso_H',  'ENet','ENet_H','Lasso_B', 'Lasso_H_B', 'ENet_B', 'ENet_H_B']
        tableA300 = pd.DataFrame(coef_100.T, columns=columns_list1)
        tableA300.to_csv(Path + '//output//' + foldlist + 'tableA3_all.csv')

    ################################define table
    columns_list = ['IS_a_50', 'OOS_a_50', 'IS_b_50', 'OOS_b_50',
                    'IS_a_100', 'OOS_a_100', 'IS_b_100', 'OOS_b_100']
    index_list = ['Oracle', 'OLS', 'OLS_H', 'PCR', 'PLS', 'Lasso', 'Lasso_H',
                  'Ridge', 'Ridge_H', 'ENet', 'ENet_H']

    tableA100 = pd.DataFrame(tableA1_all, columns=columns_list, index=index_list)
    tableA100.to_csv(Path + '//output//tableA1_all.csv')
