'''
This program is used to generate table A.4:Comparison of Average Variable Importance in Simulations
    Input file:
        * F:\MCS\output\SimuData_p50tableA3_Tree.csv
        * F:\MCS\output\SimuData_p100tableA3_Tree.csv
    Ouput file:
        F:\MCS\output\tableA4_final.csv
'''

import pandas as pd
import numpy as np

Path = 'F:\\MCS'
Data50_fold = 'SimuData_p50'
Data100_fold = 'SimuData_p100'

########################################Tree Models
foldlist = Data50_fold
P100_coef = pd.read_csv(Path + '//output//' + foldlist + 'tableA3_Tree.csv')

foldlist = Data100_fold
P200_coef = pd.read_csv(Path + '//output//' + foldlist + 'tableA3_Tree.csv')

P100_coef = P100_coef.iloc[:,1:]
P200_coef = P200_coef.iloc[:,1:]


P100_coef_temp = np.zeros(shape=(7,6))
P100_coef_temp[:3] = P100_coef.iloc[:3,:]
P100_coef_temp[3:6] = P100_coef.iloc[50:3+50,:]
P100_coef_temp1 = np.zeros(shape=(94,6))
P100_coef_temp1[:47] = P100_coef.iloc[3:50,:]
P100_coef_temp1[47:] = P100_coef.iloc[53:,:]
P100_coef_noise = np.mean(P100_coef_temp1,axis=0)
P100_coef_temp[6] = P100_coef_noise.copy()

P200_coef_temp = np.zeros(shape=(7,6))
P200_coef_temp[:3] = P200_coef.iloc[:3,:]
P200_coef_temp[3:6] = P200_coef.iloc[100:3+100,:]

P200_coef_temp1 = np.zeros(shape=(200-6,6))
P200_coef_temp1[:97] = P200_coef.iloc[3:100,:]
P200_coef_temp1[97:] = P200_coef.iloc[103:,:]
P200_coef_noise = np.mean(P200_coef_temp1,axis=0)
P200_coef_temp[6] = P200_coef_noise.copy()



column_list = ['Parameter','Method','c1','c2','c3','c1x','c2x','c3x','Noise']
index_list = ['RF', 'GBRT', 'GBRT_H','NN1','NN2','NN3','NN4','NN5']
index_temp = []
for i in range(8):
    index_temp.append(index_list[i] + '_B')

index_list = index_list + index_temp

TableA4 = pd.DataFrame(columns=column_list,index=range(32))

num = 8
TableA4.loc[0:8,'Parameter'] = 'P=50'
TableA4.loc[8:16,'Parameter'] = 'P=100'
TableA4.loc[16:24,'Parameter'] = 'P=50'
TableA4.loc[24:32,'Parameter'] = 'P=100'

TableA4.loc[0:15,'Method'] = index_list
TableA4.loc[16:,'Method'] = index_list



TableA4.iloc[:3,2:] = P100_coef_temp[:,:3].T
TableA4.iloc[8:11,2:] = P200_coef_temp[:,:3].T

TableA4.iloc[16:19,2:] = P100_coef_temp[:,3:].T
TableA4.iloc[24:27,2:] = P200_coef_temp[:,3:].T

TableA4.to_csv(Path + '//output//tableA4_Tree.csv')
