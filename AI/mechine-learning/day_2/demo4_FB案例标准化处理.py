#!/usr/bin/env python
# coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def knncls():
    """K-近邻算法预测用户查询业务"""
    data = pd.read_csv('/home/jhonchen/scikit_learn_data/FBlocation/train_0.csv')
    # print(data)
    # 1.缩小数据范围
    # data = data.query("x >1.0 & x < 1.25 & y > 2.5 & y< 2.75")

    # 2.进行标准化


    # 3.把签到位置小于N个人的删除
    

    # 4.分割数据集到训练集和测试集
    # 取出特征值和目标值
    y = data[['place_id']]

    x = data[['x', 'y', 'accuracy', 'time']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # 5.利用k-近邻算法去进行训练预测
    knn = KNeighborsClassifier(n_neighbors=5)
    # 调用fit和predict或者score
    knn.fit(x_train, y_train)
    # 预测测试集的目标值（签到位置）
    y_predict = knn.predict(x_test)
    
    print("K近邻算法预测的这些时间的业务类型：", y_predict)
    print("K近邻预测的准确率：", knn.score(x_test, y_test))
    return None




if __name__ == '__main__':
    knncls()










