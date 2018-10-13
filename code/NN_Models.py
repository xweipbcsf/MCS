'''
This code is used to get NN1-5 models result
1. Table A.1: Comparison of Predictive R2s for Machine Learning Algorithms in Simulations
  # NN Models: NN1-5
  # PC: 50 & 100
  # Model: A & B
2. Table A.4: Comparison of Average Variable Importance in Simulations
  # NN Models: NN1-5
  # PC: 50 & 100
  # Model: A & B
'''
import pickle
import random as rn
import os
import numpy as np
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)
from sklearn.metrics import r2_score
import tensorflow as tf
from keras import backend as K

K.clear_session()
tf.reset_default_graph()
tf.set_random_seed(1234)

import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import initializers

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


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


tableA1_all = np.zeros(shape=(5, 8))

for foldnum, foldlist in enumerate([Data50_fold, Data100_fold]):

    tableA1_100 = np.zeros(shape=(5, 4))

    for x_num in range(100):
        print('This is round:', x_num)

        full_Path_x = Path + '\\MCSdata\\' + foldlist + '\\' + x_file_list[x_num]
        full_Path_y1 = Path + '\\MCSdata\\' + foldlist + '\\' + r1_file_list[x_num]
        full_Path_y2 = Path + '\\MCSdata\\' + foldlist + '\\' + r2_file_list[x_num]
        X_raw = pd.read_csv(full_Path_x + '.csv', header=None)
        Y1_raw = pd.read_csv(full_Path_y1 + '.csv', header=None)
        Y2_raw = pd.read_csv(full_Path_y2 + '.csv', header=None)
        X_array = np.array(X_raw) * 100
        Y1_array = np.array(Y1_raw) * 100
        Y2_array = np.array(Y2_raw) * 100

        # define variables to store our result
        NN_result = np.zeros(shape=(5, 4))

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
        model_name = 'A'
        ################################NN1

        # ll_float = 0.9
        # learn_rate_float = 0.001
        # beta_1_float = 0.9
        # beta_2_float = 0.9
        # epsilon_float = 1e-08
        # batch_size_num = 50
        # epochs_num = 300

        space = {'ll_float': hp.uniform('ll_float', 0.8, 0.99),
                 'lr': hp.uniform('lr', 0.0009, 0.002),
                 'beta_1_float': hp.uniform('beta_1_float', 0.85, 0.95),
                 'beta_2_float': hp.uniform('beta_2_float', 0.9, 0.99),
                 'epsilon_float': hp.choice('epsilon_float', [1e-08,1e-07,1e-09]),  ##note
                 'batch_size': hp.choice('batch_size', [50, 150, 100, 200]),
                 'epochs': hp.quniform('epochs', 500, 600, 1)
                 }

        ## set params random search time,when set 50,something will be wrong
        try_num1 = int(10)

        # ################################NN1 Model
        ## model structure
        def f_NN1(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN1 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN1.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN1.add(BatchNormalization())

            model_NN1.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN1.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN1.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)

            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro}


        trials = Trials()
        best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro}

        file = open(Path + '\\model\\NN1\\' + model_name + str(x_num) + foldlist + 'Model_NN1_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        print('NN1 Model a IS:', IS_score * 100)
        print('NN1 Model a OOS:', OOS_score * 100)
        NN_result[0, 0] = IS_score
        NN_result[0, 1] = OOS_score

        K.clear_session()
        tf.reset_default_graph()

        # ################################NN2 Model

        def f_NN2(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN2 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN2.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN2.add(BatchNormalization())
            model_NN2.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))
            model_NN2.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN2.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN2.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)


            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()


            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro,
                    'layer2': layer2_W_pro,'layer3':layer3_W_pro}


        trials = Trials()
        best = fmin(f_NN2, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        print('NN2 Model a IS:', IS_score * 100)
        print('NN2 Model a OOS:', OOS_score * 100)
        NN_result[1, 0] = IS_score
        NN_result[1, 1] = OOS_score

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro':layer3_W_pro}

        file = open(Path + '\\model\\NN2\\' + model_name + str(x_num) + foldlist + 'Model_NN2_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        K.clear_session()
        tf.reset_default_graph()


        # ################################NN3 Model

        def f_NN3(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN3 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN3.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN3.add(BatchNormalization())
            model_NN3.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))
            model_NN3.add(Dense(8,
                                activation='relu',
                                kernel_initializer=init))
            model_NN3.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN3.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN3.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)


            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()
            layer4 = best_model.layers[4]
            layer4_W_pro = layer4.get_weights()


            train_predict = best_model.predict(X_traindata1, verbose=0)
            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro,
                    'layer3': layer3_W_pro, 'layer4': layer4_W_pro}


        trials = Trials()
        best = fmin(f_NN3, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        print('NN3 Model a IS:', IS_score * 100)
        print('NN3 Model a OOS:', OOS_score * 100)
        NN_result[2, 0] = IS_score
        NN_result[2, 1] = OOS_score

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layer4_W_pro = best_results['layer4']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro': layer3_W_pro,
                         'layer4_W_pro': layer4_W_pro}

        file = open(Path + '\\model\\NN3\\' + model_name + str(x_num) + foldlist + 'Model_NN3_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        K.clear_session()
        tf.reset_default_graph()



        # ################################NN4 Model

        def f_NN4(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN4 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN4.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN4.add(BatchNormalization())
            model_NN4.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))

            model_NN4.add(Dense(8,
                                activation='relu',
                                kernel_initializer=init))

            model_NN4.add(Dense(4,
                                activation='relu',
                                kernel_initializer=init))

            model_NN4.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN4.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN4.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)


            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()
            layer4 = best_model.layers[4]
            layer4_W_pro = layer4.get_weights()
            layer5 = best_model.layers[5]
            layer5_W_pro = layer5.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro,
                    'layer3': layer3_W_pro, 'layer4': layer4_W_pro, 'layer5': layer5_W_pro}


        trials = Trials()
        best = fmin(f_NN4, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        print('NN4 Model a IS:', IS_score * 100)
        print('NN4 Model a OOS:', OOS_score * 100)
        NN_result[3, 0] = IS_score
        NN_result[3, 1] = OOS_score

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layer4_W_pro = best_results['layer4']
        layer5_W_pro = best_results['layer5']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro': layer3_W_pro,
                         'layer4_W_pro': layer4_W_pro,
                         'layer5_W_pro': layer5_W_pro}

        file = open(Path + '\\model\\NN4\\' + model_name + str(x_num) + foldlist + 'Model_NN4_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        K.clear_session()
        tf.reset_default_graph()

        ####################################NN5

        def f_NN5(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN5 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN5.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN5.add(BatchNormalization())
            model_NN5.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(8,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(4,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(2,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN5.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN5.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)


            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()
            layer4 = best_model.layers[4]
            layer4_W_pro = layer4.get_weights()
            layer5 = best_model.layers[5]
            layer5_W_pro = layer5.get_weights()
            layer6 = best_model.layers[6]
            layer6_W_pro = layer6.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro,
                    'layer3': layer3_W_pro, 'layer4': layer4_W_pro,
                    'layer5': layer5_W_pro, 'layer6': layer6_W_pro}


        trials = Trials()
        best = fmin(f_NN5, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layer4_W_pro = best_results['layer4']
        layer5_W_pro = best_results['layer5']
        layer6_W_pro = best_results['layer6']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro': layer3_W_pro,
                         'layer4_W_pro': layer4_W_pro,
                         'layer5_W_pro': layer5_W_pro,
                         'layer6_W_pro': layer6_W_pro,}

        file = open(Path + '\\model\\NN5\\' + model_name + str(x_num) + foldlist + 'Model_NN5_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        print('NN5 Model a IS:', IS_score * 100)
        print('NN5 Model a OOS:', OOS_score * 100)
        NN_result[4, 0] = IS_score
        NN_result[4, 1] = OOS_score

        K.clear_session()
        tf.reset_default_graph()

        model_name = 'B'
        ###############################################################################Model B
        X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata, mean_Ytrain, X_traindata1, Y_traindata1, \
        Y_traindata_demean, Y_vdata_demean, Y_testdata_demean, Y_traindata1_demean = split_data(df_X=X_array,
                                                                                                df_Y=Y2_array)

        # ################################NN1 Model
        ## model structure
        def f_NN1(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN1 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN1.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN1.add(BatchNormalization())
            model_NN1.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN1.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN1.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)


            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro}


        trials = Trials()
        best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']
        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro}

        file = open(Path + '\\model\\NN1\\' + model_name + str(x_num) + foldlist + 'Model_NN1_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        print('NN1 model b IS:', IS_score * 100)
        print('NN1 model b OOS:', OOS_score * 100)
        NN_result[0, 2] = IS_score
        NN_result[0, 3] = OOS_score

        K.clear_session()
        tf.reset_default_graph()

        # ################################NN2 Model

        def f_NN2(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN2 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN2.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN2.add(BatchNormalization())
            model_NN2.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))
            model_NN2.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN2.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN2.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)


            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)


            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                'train_score': train_score, 'test_score': test_score,
                'layer0':layer0_W_pro, 'layer1': layer1_W_pro,
                    'layer2': layer2_W_pro,'layer3':layer3_W_pro}


        trials = Trials()
        best = fmin(f_NN2, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        print('NN2 model b IS:', IS_score * 100)
        print('NN2 model b OOS:', OOS_score * 100)
        NN_result[1, 2] = IS_score
        NN_result[1, 3] = OOS_score

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro':layer3_W_pro}

        file = open(Path + '\\model\\NN2\\' + model_name + str(x_num) + foldlist + 'Model_NN2_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        K.clear_session()
        tf.reset_default_graph()


        # ################################NN3 Model

        def f_NN3(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN3 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN3.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN3.add(BatchNormalization())
            model_NN3.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))
            model_NN3.add(Dense(8,
                                activation='relu',
                                kernel_initializer=init))
            model_NN3.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN3.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN3.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)

            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()
            layer4 = best_model.layers[4]
            layer4_W_pro = layer4.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)
            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)

            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                    'train_score': train_score, 'test_score': test_score,
                    'layer0': layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro,
                    'layer3': layer3_W_pro, 'layer4': layer4_W_pro}


        trials = Trials()
        best = fmin(f_NN3, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layer4_W_pro = best_results['layer4']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro': layer3_W_pro,
                         'layer4_W_pro': layer4_W_pro}

        file = open(Path + '\\model\\NN3\\' + model_name + str(x_num) + foldlist + 'Model_NN3_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']
        print('NN3 model b IS:', IS_score * 100)
        print('NN3 model b OOS:', OOS_score * 100)
        NN_result[2, 2] = IS_score
        NN_result[2, 3] = OOS_score

        K.clear_session()
        tf.reset_default_graph()


        # ################################NN4 Model

        def f_NN4(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN4 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN4.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN4.add(BatchNormalization())
            model_NN4.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))

            model_NN4.add(Dense(8,
                                activation='relu',
                                kernel_initializer=init))

            model_NN4.add(Dense(4,
                                activation='relu',
                                kernel_initializer=init))

            model_NN4.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN4.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN4.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)

            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()
            layer4 = best_model.layers[4]
            layer4_W_pro = layer4.get_weights()
            layer5 = best_model.layers[5]
            layer5_W_pro = layer5.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)

            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                    'train_score': train_score, 'test_score': test_score,
                    'layer0': layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro,
                    'layer3': layer3_W_pro, 'layer4': layer4_W_pro, 'layer5': layer5_W_pro}


        trials = Trials()
        best = fmin(f_NN4, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]

        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']


        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layer4_W_pro = best_results['layer4']
        layer5_W_pro = best_results['layer4']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro': layer3_W_pro,
                         'layer4_W_pro': layer4_W_pro,
                         'layer5_W_pro': layer5_W_pro}

        file = open(Path + '\\model\\NN4\\' + model_name + str(x_num) + foldlist + 'Model_NN4_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()


        print('NN4 model b IS:', IS_score * 100)
        print('NN4 model b OOS:', OOS_score * 100)
        NN_result[3, 2] = IS_score
        NN_result[3, 3] = OOS_score

        K.clear_session()
        tf.reset_default_graph()

        ####################################NN5

        def f_NN5(params):
            ## define params
            ll_float = params["ll_float"]  # 0.1
            learn_rate_float = params["lr"]  # 0.01
            beta_1_float = params["beta_1_float"]  # 0.9
            beta_2_float = params["beta_2_float"]  # 0.999
            epsilon_float = params["epsilon_float"]  # 1e-08
            batch_size_num = params['batch_size']  #
            epochs_num = params['epochs']  # 50
            #define NN1
            model_NN5 = Sequential()
            init = initializers.he_normal(seed=100)
            model_NN5.add(Dense(32, input_dim=len(X_traindata[0]),
                                activation='relu',
                                kernel_initializer=init,
                                kernel_regularizer=regularizers.l1(ll_float)))
            model_NN5.add(BatchNormalization())
            model_NN5.add(Dense(16,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(8,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(4,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(2,
                                activation='relu',
                                kernel_initializer=init))

            model_NN5.add(Dense(1))

            ## comile model
            adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
            model_NN5.compile(loss='mse', optimizer=adam)

            ## callback fun
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                           patience=3, verbose=0, mode='auto')
            model_filepath = Path + '\\model\\best_weights.h5'
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                         monitor='val_loss', mode='min', save_best_only='True')
            callback_lists = [early_stopping, checkpoint]

            ## fit model
            model_NN5.fit(X_traindata, Y_traindata_demean,
                          batch_size=int(batch_size_num),
                          epochs=int(epochs_num),
                          verbose=0,
                          validation_data=(X_vdata, Y_vdata_demean),
                          callbacks=callback_lists,
                          shuffle=False)

            ##get the best model
            best_model = load_model(model_filepath)

            layer0 = best_model.layers[0]
            layer0_W_pro = layer0.get_weights()
            layer1 = best_model.layers[1]
            layer1_W_pro = layer1.get_weights()
            layer2 = best_model.layers[2]
            layer2_W_pro = layer2.get_weights()
            layer3 = best_model.layers[3]
            layer3_W_pro = layer3.get_weights()
            layer4 = best_model.layers[4]
            layer4_W_pro = layer4.get_weights()
            layer5 = best_model.layers[5]
            layer5_W_pro = layer5.get_weights()
            layer6 = best_model.layers[6]
            layer6_W_pro = layer6.get_weights()

            train_predict = best_model.predict(X_traindata1, verbose=0)

            Y_pre_tlist = []
            for x in train_predict[:, 0]:
                Y_pre_tlist.append(x + mean_Ytrain)

            train_score = r2_score(Y_traindata1, Y_pre_tlist)

            Y_pre_v = best_model.predict(X_vdata, verbose=0)

            Y_pre_vlist = []
            for x in Y_pre_v[:, 0]:
                Y_pre_vlist.append(x + mean_Ytrain)

            v_score = 1 - np.sum((Y_vdata - np.array(Y_pre_vlist).reshape(-1,1)) ** 2) / np.sum((Y_vdata - mean_Ytrain) ** 2)

            ## prediction & save
            Y_pre = best_model.predict(X_testdata, verbose=0)
            Y_pre_list = []
            for x in Y_pre[:, 0]:
                Y_pre_list.append(x + mean_Ytrain)
            test_score = 1 - np.sum((Y_testdata - np.array(Y_pre_list).reshape(-1,1)) ** 2) / np.sum((Y_testdata - mean_Ytrain) ** 2)

            K.clear_session()

            return {'loss': -v_score, 'status': STATUS_OK,
                    'train_score': train_score, 'test_score': test_score,
                    'layer0': layer0_W_pro, 'layer1': layer1_W_pro, 'layer2': layer2_W_pro,
                    'layer3': layer3_W_pro, 'layer4': layer4_W_pro,
                    'layer5': layer5_W_pro, 'layer6': layer6_W_pro}


        trials = Trials()
        best = fmin(f_NN5, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
        loss_list = trials.losses()
        min_loss = min(loss_list)
        for k in range(try_num1):
            if min_loss == loss_list[k]:
                key = k
        best_results = trials.results[key]


        layer0_W_pro = best_results['layer0']
        layer1_W_pro = best_results['layer1']
        layer2_W_pro = best_results['layer2']
        layer3_W_pro = best_results['layer3']
        layer4_W_pro = best_results['layer4']
        layer5_W_pro = best_results['layer5']
        layer6_W_pro = best_results['layer6']
        layers_weight = {'layer0_W_pro': layer0_W_pro,
                         'layer1_W_pro': layer1_W_pro,
                         'layer2_W_pro': layer2_W_pro,
                         'layer3_W_pro': layer3_W_pro,
                         'layer4_W_pro': layer4_W_pro,
                         'layer5_W_pro': layer5_W_pro,
                         'layer6_W_pro': layer6_W_pro}

        file = open(Path + '\\model\\NN5\\' + model_name + str(x_num) + foldlist + 'Model_NN5_weight.pkl', 'wb')
        pickle.dump(layers_weight, file)
        file.close()


        IS_score = best_results['train_score']
        OOS_score = best_results['test_score']

        print('NN5 model b IS:', IS_score * 100)
        print('NN5 model b OOS:', OOS_score * 100)
        NN_result[4, 2] = IS_score
        NN_result[4, 3] = OOS_score

        K.clear_session()
        tf.reset_default_graph()

        tableA1_100 = tableA1_100 + NN_result

    tableA1_all[:, 4 * foldnum:4 * (foldnum + 1)] = tableA1_100


################################define table
columns_list = ['IS_a_50', 'OOS_a_50', 'IS_b_50', 'OOS_b_50',
                'IS_a_100', 'OOS_a_100', 'IS_b_100', 'OOS_b_100']
index_list = ['NN1','NN2','NN3','NN4','NN5']

tableA100 = pd.DataFrame(tableA1_all, columns=columns_list, index=index_list)
tableA100.to_csv(Path + '//output//tableA1_NNs.csv')
