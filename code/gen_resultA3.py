'''
This program is used to generate table A.3:Comparison of Average Variable Selection Frequencies in Simulations
    Input file:
        * F:\MCS\output\SimuData_p50tableA3_all.csv
        * F:\MCS\output\SimuData_p100tableA3_all.csv
    Ouput file:
        F:\MCS\output\tableA3_final.csv
'''
import pandas as pd
import numpy as np

Path = 'F:\\MCS'
Data50_fold = 'SimuData_p50'
Data100_fold = 'SimuData_p100'

foldlist = Data50_fold
P100_coef = pd.read_csv(Path + '//output//' + foldlist + 'tableA3_all.csv')


foldlist = Data100_fold
P200_coef = pd.read_csv(Path + '//output//' + foldlist + 'tableA3_all.csv')

P100_coef = P100_coef.iloc[:, 1:]
P200_coef = P200_coef.iloc[:, 1:]

P100_coef_temp = np.zeros(shape=(7, 8))
P100_coef_temp[:3] = P100_coef.iloc[:3, :]
P100_coef_temp[3:6] = P100_coef.iloc[50:3 + 50, :]
P100_coef_temp1 = np.zeros(shape=(94, 8))
P100_coef_temp1[:47] = P100_coef.iloc[3:50, :]
P100_coef_temp1[47:] = P100_coef.iloc[53:, :]
P100_coef_noise = np.mean(P100_coef_temp1, axis=0)
P100_coef_temp[6] = P100_coef_noise.copy()

P200_coef_temp = np.zeros(shape=(7, 8))
P200_coef_temp[:3] = P200_coef.iloc[:3, :]
P200_coef_temp[3:6] = P200_coef.iloc[100:3 + 100, :]

P200_coef_temp1 = np.zeros(shape=(200 - 6, 8))
P200_coef_temp1[:97] = P200_coef.iloc[3:100, :]
P200_coef_temp1[97:] = P200_coef.iloc[103:, :]
P200_coef_noise = np.mean(P200_coef_temp1, axis=0)
P200_coef_temp[6] = P200_coef_noise.copy()

column_list = ['Parameter', 'Method', 'c1', 'c2', 'c3', 'c1x', 'c2x', 'c3x', 'Noise']
index_list = ['Lasso', 'Lasso_H', 'ENet', 'ENet_H', 'Lasso_B', 'Lasso_H_B', 'ENet_B', 'ENet_H_B']
TableA3 = pd.DataFrame(columns=column_list, index=range(16))

TableA3.loc[0:4, 'Parameter'] = 'P=50'
TableA3.loc[4:8, 'Parameter'] = 'P=100'
TableA3.loc[8:12, 'Parameter'] = 'P=50'
TableA3.loc[12:16, 'Parameter'] = 'P=100'

TableA3.loc[:3, 'Method'] = index_list[:4]
TableA3.loc[4:7, 'Method'] = index_list[:4]
TableA3.loc[8:11, 'Method'] = index_list[4:]
TableA3.loc[12:15, 'Method'] = index_list[4:]

TableA3.iloc[:4, 2:] = P100_coef_temp[:, :4].T
TableA3.iloc[4:8, 2:] = P200_coef_temp[:, :4].T

TableA3.iloc[8:12, 2:] = P100_coef_temp[:, 4:].T
TableA3.iloc[12:16, 2:] = P200_coef_temp[:, 4:].T

for x in TableA3.columns[2:]:
    TableA3[x] = TableA3[x]/100

TableA3.to_csv(Path + '//output//tableA3_final.csv')
