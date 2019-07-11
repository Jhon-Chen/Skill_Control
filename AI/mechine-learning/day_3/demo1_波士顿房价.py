from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def my_linear():
    """线性回归的两种求解方式来进行房价预测"""
    # 获取数据进行数据集分割
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.3)

    # 对数据进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 使用线性回归的模型进行训练预测
    # 正规方程求解方式 LinearRegression
    lr = LinearRegression(fit_intercept=True)

    # fit之后已经得出正规方程的结果
    lr.fit(x_train, y_train)
    print("正规方程计算出的权重(斜率)：", lr.coef_)
    print("正规方程计算出的偏置(截距)：", lr.intercept_)

    # 调用predict去预测目标值
    y_predict = lr.predict(x_test)
    print("测试集的预测结果是：", y_predict[:50])

    # 利用均方误差来评估回归性能
    err = mean_squared_error(y_test, y_predict)
    print("回归算法的误差的平方为：", err)

    # 使用SGD梯度下降方法进行预测
    # sgd = SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate="invscaling")
    # 尝试着修改学习率
    sgd = SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate="constant", eta0=0.0001)
    sgd.fit(x_train, y_train)
    print("梯度下降计算出的权重：", sgd.coef_)
    print("梯度下降计算出的偏置：", sgd.intercept_)

    y_predict_sgd = sgd.predict(x_test)
    err_sgd = mean_squared_error(y_test, y_predict_sgd)
    print("梯度下降法的预测误差的平方：", err_sgd)

    # 使用带有L2正则化的线性回归去预测
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print("岭回归计算出的权重：", rd.coef_)
    print("岭回归计算出的偏置：", rd.intercept_)

    y_predict_rd = rd.predict(x_test)
    err_rd = mean_squared_error(y_test, y_predict_rd)
    print("梯度下降法的预测误差的平方：", err_rd)

    return None


if __name__ == '__main__':
    my_linear()
