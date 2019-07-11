from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


def logistic():
    """使用逻辑回归进行肿瘤数据预测"""
    # 因为他数据中没有列标签，所以需要自己导入
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    # 读取数据，处理缺失值
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                       names=column_name)
    # print(data)
    # 处理缺失值（将？替换成np.nan）
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()
    print(data.shape)

    # 取出特征目标值，分割数据集
    x = data.iloc[:, 1:10]
    y = data.iloc[:, 10]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 进行标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 逻辑回归进行训练与预测
    log = LogisticRegression()
    # 默认会把4（恶性）当做正例
    log.fit(x_train, y_train)
    print("逻辑回归的权重:", log.coef_)
    print("逻辑回归的偏置:", log.intercept_)

    print("逻辑回归在测试集当中的预测类别:", log.predict(x_test))
    print("逻辑回归的准确率:", log.score(x_test, y_test))

    # 召回率
    print("召回率为:", classification_report(y_test, log.predict(x_test), labels=[2, 4], target_names=['良性', '恶性']))

    # 查看分类在这些数据中的AUC指标值
    y_test = np.where(y_test > 2.5, 1, 0)
    print("此场景的分类器的AUC指标为:", roc_auc_score(y_test, log.predict(x_test)))

    return None


if __name__ == '__main__':
    logistic()