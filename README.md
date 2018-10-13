# Monte Carlo Simulation复现程序说明
## 项目说明
本文档为复现手册的附录四，用于记录复现文献中的附录A：Monte Carlo Simulation结果中遇到的问题及相关复现程序信息。
## 项目数据
项目数据有两个来源：
*	自己通过文章给出数据生成方式自己使用python编程生成数据，具体方法见下文，实现文档为Data_creat_final100.py
*	论文作者Dacheng老师给了我们Matlab版本Simulation的code，我们可以通过Matlab的code生成所需数据，实现文档为DGP.m
## 项目程序
项目程序主要由三部分组成：数据生成，模型构建，结果整理。其中，
*	数据生成用于提供程序所需数据，程序为Data_creat_final100.py 
*	模型构建用于提供各个模型的结果呈现，包括线性、树、神经网络三个模型：
*	线性模型主要涉及11个不同的线性模型构建，其中包括以下几种：OLS(H)\PCR\PLS\Lasso(H)\Ridge(H)\ENet(H)\Oracle，程序为Liner_Models.py
*	决策树类模型主要包括RF\GBRT(H)，程序为Tree_Models.py
*	神经网络模型主要包括NN1-5，程序为NN_Models.py
*	结果整理用于整理结果，制作表格，其中包括：
*	计算神经网络各个因子重要性的计算，程序为NN_VIP.py
*	制作表格A.3，程序为gen_resultA3.py
*	制作表格A.4，程序为gen_resultA4.py
## 程序说明
*	请确保自己的文件夹结构与我的一致
*	确保安装了keara\tensorflow\hyperopt库
*	感谢Xiu Dacheng老师提供了Matlab版本的程序给我们学习，本程序只是个人在学习论文的过程中自己编写的程序，文章的一切解释权利均为原论文团队所有，程序中的一切错误均为本人水平有限导致，与原文无关
## 其他信息
* 版本号：V2
* 语言环境：Python                  
* 项目开始时间：2018年9月25日
* 项目成员：吴辉航、赵辉、魏行空、张欣然
* 文献信息：Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, Empirical Asset Pricing via Machine Learning (June 11, 2018). Chicago Booth Research Paper No. 18-04. Available at SSRN: https://ssrn.com/abstract=3159577.
## 备注
