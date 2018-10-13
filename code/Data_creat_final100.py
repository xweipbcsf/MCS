import pandas as pd
import numpy as np
import pickle
import math
from scipy.stats import norm
from scipy.stats import t as t_norm
from scipy.stats import uniform


Path = 'F:\\MCS'
Data50_fold = 'SimuData_p50'
Data100_fold = 'SimuData_p100'
fold_list = [Data50_fold, Data100_fold]
x_file_list = []
r1_file_list = []
r2_file_list = []
for i in range(1, 101, 1):
    x_file_list.append('c' + str(i))
    r1_file_list.append('r1_' + str(i))
    r2_file_list.append('r2_' + str(i))

pc_num = 0

for pc1 in [50,100]:

    for m in range(100):
        # 0 set size & df
        T_num = 180
        id_num = 200
        Xt_list = ['x1']
        temp = list(range(1, pc1 + 1))
        ct_list = []
        for x in temp:
            ct_list.append('c' + str(x))
        other_columns_name = ['gz1', 'gz2', 'e', 'ret1','ret2']
        columns_name = Xt_list + ct_list + other_columns_name
        final_data = pd.DataFrame(columns=columns_name,index=range(T_num*id_num))

        # 1 simulate characteristics Cij,t
        data_low = 0.9
        data_scale = 0.1
        data_size = pc1
        pj = uniform.rvs(loc=data_low, scale=data_scale, size=data_size)
        data_mean = 0
        data_std = 1
        data_size = id_num

        # epsilon_ij_t = norm.rvs(loc=data_mean, scale=data_std, size=data_size)

        c = np.zeros(shape=(id_num * T_num, pc1))

        for j in range(pc1):
            c[0:200,j] = norm.rvs(loc=data_mean, scale=data_std, size=data_size)
            for t in range(1, T_num):
                c[200 * t:200 * (t + 1), j] = c[200 * (t - 1):200 * t, j] * pj[j] + norm.rvs(loc=data_mean, scale=data_std,
                                                                                             size=data_size) * np.sqrt(1 - pj[j] ** 2)
        c_rank = np.zeros(shape=(id_num * T_num, pc1))

        # rank over cross-section

        for j in range(pc1):
            temp_series = pd.Series(c[:, j])
            temp_series = temp_series.rank()
            temp_series = 2 * temp_series / (len(temp_series) + 1) - 1
            c_rank[:, j] = temp_series.copy()

        final_data_100 = final_data
        final_data_100.loc[:, ct_list] = c_rank

        ##2 e
        data_mean = 0
        data_std = 0.05
        data_size = T_num
        vt1_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size)
        vt2_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size)
        vt3_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size)
        temp_array = np.zeros(shape=[3, T_num])

        temp_array[0, :] = vt1_array
        temp_array[1, :] = vt2_array
        temp_array[2, :] = vt3_array

        beta = c_rank[:, :3].copy()
        beta_v = np.zeros(shape=(id_num*T_num,3))

        for i in range(T_num):
            for j in range(3):
                beta_v[200 * i:200 * (i + 1), j] = beta[200 * i:200 * (i + 1), j] * temp_array[j, i]

        data_fr = 5
        data_mean = 0
        data_scale = 0.05
        data_size = id_num*T_num

        epsilon2_it = t_norm.rvs(df=data_fr, loc=data_mean, scale=data_scale, size=data_size)
        e = beta_v[:, 0] + beta_v[:, 1] + beta_v[:, 2] + epsilon2_it
        final_data_100['e'] = e


        ##3 xt
        p = 0.95
        data_mean = 0
        data_std = math.sqrt(1 - 0.95 * 0.95)
        data_size = 1
        #u1_t_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size)

        x1 = np.zeros(shape=T_num * id_num)
        x1[:200] = np.random.randn(1)

        for t in range(1,T_num):
            x1[200 * t:200 * (t + 1)] = p * x1[200 * (t - 1):200 * t]  + np.random.randn(1)*np.sqrt(1-p**2)

        # save x1
        final_data_100['x1'] = x1
        ## model A: gz1

        seita1 = np.array([[0.02, 0.02, 0.02]]).T
        temp = final_data_100['x1'] * final_data_100['c3']
        c13 = np.zeros(shape=(id_num*T_num,3))
        c13[:, 0] = np.array(final_data_100['c1'])
        c13[:, 1] = np.array(final_data_100['c2'])
        c13[:, 2] = np.array(temp)

        gz1 = np.dot(c13, seita1)
        final_data_100['gz1'] = gz1

        seita2 = np.array([[0.04, 0.03, 0.012]]).T
        temp_c1 = final_data_100['c1'] ** 2
        temp_c2 = final_data_100['c1'] * final_data_100['c2']
        temp_c3 = np.sign(final_data_100['x1'] * final_data_100['c3'])

        ## model A: gz2
        c23 =  np.zeros(shape=(id_num*T_num,3))
        c23[:, 0] = np.array(temp_c1)
        c23[:, 1] = np.array(temp_c2)
        c23[:, 2] = np.array(temp_c3)

        gz2 = np.dot(c23, seita2)
        final_data_100['gz2'] = gz2

        final_data_100['ret1'] = final_data_100['e'] + final_data_100['gz1']
        final_data_100['ret2'] = final_data_100['e'] + final_data_100['gz2']

        final_data_100_x1 = np.zeros(shape=(id_num*T_num,2*pc1))
        final_data_100_x1[:,:pc1] = final_data_100.loc[:,ct_list]


        for i in range(pc1):
            final_data_100.iloc[:,i+1] = final_data_100.iloc[:,i+1]*final_data_100.loc[:,'x1']


        final_data_100_x1[:,pc1:pc1*2] = final_data_100.loc[:,ct_list]
        final_data_100_y1 = final_data_100['ret1']
        final_data_100_y2 = final_data_100['ret2']

        file = open(Path + '\\data\\' + fold_list[pc_num] + '\\' + x_file_list[m] + '.pkl', 'wb')
        pickle.dump(final_data_100_x1, file)
        file.close()
        file = open(Path + '\\data\\' + fold_list[pc_num] + '\\' + r1_file_list[m] + '.pkl', 'wb')
        pickle.dump(final_data_100_y1, file)
        file.close()
        file = open(Path + '\\data\\' + fold_list[pc_num] + '\\' + r2_file_list[m] + '.pkl', 'wb')
        pickle.dump(final_data_100_y2, file)
        file.close()

    pc_num = pc_num+1


#
# ## check some data
# from sklearn.metrics import r2_score
# import statsmodels.api as sm
#
# ## creat data
# gz1 = np.array(final_data_100_y['gz1'])
# ret1 = np.array(final_data_100_y['ret1'])
# gz2 = np.array(final_data_100_y['gz2'])
# ret2 = np.array(final_data_100_y['ret2'])
#
# ## return volatility 30%
# vol1 = np.zeros(180)
# for i in range(180):
#     vol1[i] = np.std(ret1[200 * i:200 * (i + 1)], ddof=1)
# print('Model A return VOL', np.mean(vol1)*math.sqrt(12))
#
#
# vol2 = np.zeros(180)
# for i in range(180):
#     vol2[i] = np.std(ret2[200 * i:200 * (i + 1)], ddof=1)
# print('Model B return VOL', np.mean(vol2)*math.sqrt(12))
#
# ## average time series R square 50%
# y = np.zeros(shape=id_num-1)
# X = np.zeros(shape=id_num-1)
#
#
#
#
# R2 = []
# for i in range(id_num):
#     temp_iret = ret1[i::199].copy()
#     X = temp_iret[:-1].copy()
#     Y = temp_iret[1:].copy()
#     X = sm.add_constant(X)
#     model = sm.OLS(Y, X)
#     results = model.fit()
#     R2.append(results.rsquared)
# print('Model A time series R2', np.mean(R2))
#
#
# R2 = []
# for i in range(id_num):
#     temp_iret = ret1[i::200].copy()
#     X = temp_iret[:-1].copy()
#     Y = temp_iret[1:].copy()
#     R2.append(r2_score(X,Y))
# print('Model A time series R2', np.mean(R2))
#
#
#
# ## Cross-Sectional R square 25%
# y = np.zeros(shape=200)
# X = np.zeros(shape=200)
#
# R2 = []
# for i in range(180):
#     X = gz1[200 * i:200 * (i + 1)].copy()
#     y = ret1[200 * i:200 * (i + 1)].copy()
#     X = sm.add_constant(X)
#     model = sm.OLS(y, X)
#     results = model.fit()
#     R2.append(results.rsquared)
# print('Model A Cross-Sectional R2', np.mean(R2))
#
#
#
#
#
#
# for i in range(180):
#     X = gz2[200 * i:200 * (i + 1)].copy()
#     y = ret2[200 * i:200 * (i + 1)].copy()
#     X = sm.add_constant(X)
#     model = sm.OLS(y, X)
#     results = model.fit()
#     R2.append(results.rsquared)
#
# print('Model B Cross-Sectional R2', np.mean(R2))
#
#
#
# print('Model A Predictive R2', r2_score(ret1,gz1))
# print('Model B Predictive R2', r2_score(ret2,gz2))
