#!/usr/bin/env python
# coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import pandas as pd

def knncls():
    """K-近邻算法预测用户查询业务"""
    data = pd.read_csv('/home/jhonchen/scikit_learn_data/FBlocation/train_0.csv')
    # 1.缩小数据范围
    # 2.数据逻辑筛选操作
    # data = data.query("x >1.0 & x < 1.25 & y > 2.5 & y< 2.75")

    # 3.取出特征值和目标值
    y = data['place_id']
    x = data[['x', 'y', 'accuracy', 'time']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 4.进行数据的标准化处理
    # 实例化
    std = StandardScaler()
    # 对训练集的特征值做标准化处理
    x_train = std.fit_transform(x_train)
    # 对测试集的特征值做标准化处理
    x_test = std.fit_transform(x_test)
    
    # 5.删除入住次数少于5次的位置
    # 需要补习pandas函数的语法等
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 5].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]


    # 6.利用k-近邻算法去进行训练预测
    # knn = KNeighborsClassifier(n_neighbors=2)
    # 调用fit和predict或者score
    # knn.fit(x_train, y_train)
    # 预测测试集的目标值（签到位置）
    # y_predict = knn.predict(x_test)
    
    # print("K近邻算法预测的这些时间的业务类型：", y_predict)
    # print("K近邻预测的准确率：", knn.score(x_test, y_test))

    # 应用网格搜索+交叉验证对k-近邻算法进行调优
    knn = KNeighborsClassifier()
    # 对knn来说，如果数据量比较大一般k=根号（样本）
    param = {"n_neighbors":[1, 2, 3, 4, 5]}
    gc = GridSearchCV(knn, param_grid=param, cv=2)

    # fit输入数据
    gc.fit(x_train, y_train)

    # 查看模型的超参数调优的过程,交叉验证的结果
    print("在2折交叉验证中最好的结果：", gc.best_score_)
    print("选择的最好的模型参数是：", gc.best_estimator_)
    print("每次交叉验证的验证集的准确率：", gc.cv_results_)

    # 预测测试集的准确率
    print("在测试集中的最终预测结果为：", gc.score(x_test, y_test))
    return None



if __name__ == '__main__':
    knncls()










