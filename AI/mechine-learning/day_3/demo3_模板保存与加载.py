from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

def my_linear():
    """线性回归的两种求解方式来进行房价预测"""
    # 获取数据进行数据集分割
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.3)

    # 对数据进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 使用带有L2正则化的线性回归去预测
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)

    # 保存模型
    # joblib.dump(rd, "my_Ridge.pkl")
    # 直接使用加载模型预测
    rd_model = joblib.load("my_Ridge.pkl")

    print("岭回归计算出的权重：", rd_model.coef_)
    print("岭回归计算出的偏置：", rd_model.intercept_)

    y_predict_rd = rd.predict(x_test)
    err_rd = mean_squared_error(y_test, y_predict_rd)
    print("梯度下降法的预测误差的平方：", err_rd)

    return None


if __name__ == '__main__':
    my_linear()
