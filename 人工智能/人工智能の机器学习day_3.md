# 人工智能の机器学习day_3

[TOC]

## 回归与聚类算法

### 线性回归

回忆一下，回归问题的判断是什么？

* *当这个问题有目标值切这个目标值是连续值的时候，就是回归的*

#### 线性回归的原理

**应用场景**

- 房价预测
- 销售额度预测
- 金融：贷款额度预测、利用线性回归以及系数分析因子

##### 什么是线性回归

**定义与公式**

线性回归(Linear regression)是利用**回归方程(函数)**对一个或**多个自变量(特征值)和因变量(目标值)之间**关系进行建模的一种分析方式。

- 特点：只有一个自变量的情况称为*单变量回归*，大于一个自变量情况的叫做*多元回归*

![image7c962.png](https://miao.su/images/2019/07/10/image7c962.png)

那么怎么理解呢？我们来看几个例子

- *期末成绩：0.7×考试成绩+0.3×平时成绩*
- *房子价格 = 0.02×中心区域的距离 + 0.04×城市一氧化氮浓度 + (-0.12×自住房平均房价) + 0.254×城镇犯罪率*

上面两个例子，*我们看到特征值与目标值之间建立的一个关系，这个可以理解为回归方程*。

**线性回归的特征与目标的关系分析**

线性回归当中的关系有两种，一种是线性关系，另一种是非线性关系。*在这里我们只能画一个平面更好去理解，所以都用单个特征举例子。*

线性关系：

![imagecda37.png](https://miao.su/images/2019/07/10/imagecda37.png)
![image41cf6.png](https://miao.su/images/2019/07/10/image41cf6.png)

*注释：如果在单特征与目标值的关系呈直线关系，或者两个特征与目标值呈现平面的关系更高维度的我们不用自己去想，记住这种关系即可*

非线性关系：

![image2831a.png](https://miao.su/images/2019/07/10/image2831a.png)

*注释：为什么会这样的关系呢？原因是什么？我们后面 讲解过拟合欠拟合重点介绍如果是非线性关系。

#### 线性回归的损失和优化原理

假设刚才的房子例子，真实的数据之间存在这样的关系：

```python
真实关系：真实房子价格 = 0.02×中心区域的距离 + 0.04×城市一氧化氮浓度 + (-0.12×自住房平均房价) + 0.254×城镇犯罪率
```

那么现在，我们随意指定一个关系（猜测）：

```python
随机指定关系：预测房子价格 = 0.25×中心区域的距离 + 0.14×城市一氧化氮浓度 + 0.42×自住房平均房价 + 0.34×城镇犯罪率
```

请问这样的话，会发生什么？真实结果与我们预测的结果之间是不是存在一定的误差呢？类似这样样子:
![image7414a.png](https://miao.su/images/2019/07/10/image7414a.png)
那么存在这个误差，我们将这个误差给衡量出来。

##### 损失函数

总损失定义为：
![imagee7ee4.png](https://miao.su/images/2019/07/10/imagee7ee4.png)

- y_i为第i个训练样本的真实值
- h(x_i)为第i个训练样本特征值组合预测函数
- 又称最小二乘法

*如何去减少这个损失，使我们预测的更加准确些？既然存在了这个损失，我们一直说机器学习有自动学习的功能，在线性回归这里更是能够体现。这里可以通过一些优化方法去优化（其实是数学当中的求导功能）回归的总损失！！！*

##### 优化算法

*如何去求模型当中的W，使得损失最小？（目的是找到最小损失对应的W值）*目的：优化权重和偏执

线性回归经常使用的两种优化算法 

- **正规方程**
  ![image342cd.png](https://miao.su/images/2019/07/10/image342cd.png)

  *理解：X为特征值矩阵，y为目标值矩阵。直接求到最好的结果
  缺点：当特征过多过复杂时，求解速度太慢并且得不到结果*

![image87fdf.png](https://miao.su/images/2019/07/10/image87fdf.png)

* **梯度下降**（Gradient Descent）

  

![image69929.png](https://miao.su/images/2019/07/10/image69929.png)

理解：α为学习速率，需要手动指定（超参数），α旁边的整体表示方向沿着这个函数下降的方向，最后就能找到山谷的最低点，然后更新W值使用：面对训练数据规模十分庞大的任务 ，能够找到较好的结果

我们通过两个图更好理解梯度下降的过程：
![image1ddad.png](https://miao.su/images/2019/07/10/image1ddad.png)

![image110f8.png](https://miao.su/images/2019/07/10/image110f8.png)

*所以有了梯度下降这样一个优化算法，回归就有了"自动学习"的能力*

优化动态图展示：
![线性回归优化动态图](C:\Users\Jhon\Pictures\markdown_file\线性回归优化动态图.gif)



##### 线性回归API

- sklearn.linear_model.LinearRegression(fit_intercept=True)
  - 通过正规方程优化
  - fit_intercept：是否计算偏置
  - LinearRegression.coef_：回归系数
  - LinearRegression.intercept_：偏置
- sklearn.linear_model.SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate ='invscaling', eta0=0.01)
  - SGDRegressor类实现了随机梯度下降学习，它支持不同的**loss函数和正则化惩罚项**来拟合线性回归模型。
  - loss:损失类型
    - **loss=”squared_loss”: 普通最小二乘法**
  - fit_intercept：是否计算偏置
  - learning_rate : string, optional
    - 学习率填充
    - **'constant': eta = eta0**
    - **'optimal': eta = 1.0 / (alpha \* (t + t0)) [default]**
    - 'invscaling': eta = eta0 / pow(t, power_t)
      - **power_t=0.25:存在父类当中**
    - **对于一个常数值的学习率来说，可以使用learning_rate=’constant’ ，并使用eta0来指定学习率。**
  - SGDRegressor.coef_：回归系数
  - SGDRegressor.intercept_：偏置

#### 波士顿房价预测

* 数据介绍
  ![imagec0644.png](https://miao.su/images/2019/07/10/imagec0644.png)

  ![image6c5b6.png](https://miao.su/images/2019/07/10/image6c5b6.png)

  *给定的这些特征，是专家们得出的影响房价的结果属性。我们此阶段不需要自己去探究特征是否有用，只需要使用这些特征。到后面量化很多特征需要我们自己去寻找。*

##### 分析

回归当中的数据大小不一致，会导致结果影响较大。所以需要做标准化处理（前面K-近邻算法也进行了标准化，贝叶斯和随机森林没有）。

* 数据集的分割与标准化处理
* 回归预测
* 线性回归的算法效果评估

##### 回归性能评估

均方误差(Mean Squared Error)（MSE)评价机制：
![imageed635.png](https://miao.su/images/2019/07/10/imageed635.png)

> 注：y^i为预测值，¯y为真实值

- sklearn.metrics.mean_squared_error(y_true, y_pred)
  - 均方误差回归损失
  - y_true:真实值
  - y_pred:预测值
  - return:浮点数结果

##### 代码

```python
from sklearn.linear_model import LinearRegression, SGDRegressor
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

    return None


if __name__ == '__main__':
    my_linear()
```

#### 正规方程与梯度下降对比

![be82a27878c0f128af511.png](https://miao.su/images/2019/07/10/be82a27878c0f128af511.png)

- 文字对比

|       梯度下降       |            正规方程             |
| :------------------: | :-----------------------------: |
|    需要选择学习率    |             不需要              |
|     需要迭代求解     |          一次运算得出           |
| 特征数量较大可以使用 | 需要计算方程，时间复杂度高O(n3) |

- 选择：
  - 小规模数据：
    - **LinearRegression(不能解决拟合问题)**
    - 岭回归
  - 大规模数据：SGDRegressor

#### 拓展——关于优化方法GD、SGD、SAG

##### GD

*梯度下降(Gradient Descent)，原始的梯度下降法需要计算所有样本的值才能够得出梯度，计算量大，所以后面才有会一系列的改进。*

##### SGD

*随机梯度下降(Stochastic gradient descent)是一个优化方法。它在一次迭代时只考虑一个训练样本。*

- SGD的优点是：
  - 高效
  - 容易实现
- SGD的缺点是：
  - SGD需要许多超参数：比如正则项参数、迭代数。
  - SGD对于特征标准化是敏感的。

##### SAG

随机平均梯度法(Stochasitc Average Gradient)，由于收敛的速度太慢，有人提出SAG等基于梯度下降的算法

> Scikit-learn： SGDRegressor、岭回归、逻辑回归等当中都会有SAG优化

#### 总结

- 线性回归的损失函数-均方误差
- 线性回归的优化方法
  - 正规方程
  - 梯度下降
- 线性回归的性能衡量方法-均方误差
- sklearn的SGDRegressor API 参数



### 过拟合与欠拟合

*问题：训练数据训练的很好啊，误差也不大，为什么在测试集上面有问题呢？*

当算法在某个数据集当中出现这种情况，可能就出现了过拟合现象。

#### 什么是过拟合和欠拟合

* 欠拟合
  ![image5121e.png](https://miao.su/images/2019/07/10/image5121e.png)
* 过拟合
  ![imagec10f6.png](https://miao.su/images/2019/07/10/imagec10f6.png)
* 分析
  - 第一种情况：因为机器学习到的天鹅特征太少了，导致区分标准太粗糙，不能准确识别出天鹅。
  - 第二种情况：机器已经基本能区别天鹅和其他动物了。然后，很不巧已有的天鹅图片全是白天鹅的，于是机器经过学习后，会认为天鹅的羽毛都是白的，以后看到羽毛是黑的天鹅就会认为那不是天鹅。

#### 定义

- 过拟合：一个假设在训练数据上能够获得比其他假设更好的拟合， 但是在测试数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。(模型过于复杂)
- 欠拟合：一个假设在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据，此时认为这个假设出现了欠拟合的现象。(模型过于简单)
  ![image7d503.png](https://miao.su/images/2019/07/10/image7d503.png)
- *那么是什么原因导致模型复杂？线性回归进行训练学习的时候变成模型会变得复杂，这里就对应前面说的线性回归的两种关系，非线性关系的数据，也就是存在很多无用的特征或者现实中的事物特征跟目标值的关系并不是简单的线性关系。*

#### 原因及解决办法

- 欠拟合原因以及解决办法
  - 原因：学习到数据的特征过少
  - 解决办法：增加数据的特征数量
- 过拟合原因以及解决办法
  - 原因：原始特征过多，存在一些嘈杂特征， 模型过于复杂是因为模型尝试去兼顾各个测试数据点
  - 解决办法：
    - 正则化

> 在这里针对回归，我们选择了正则化。但是对于其他机器学习算法如分类算法来说也会出现这样的问题，除了一些算法本身作用之外（决策树、神经网络），我们更多的也是去自己做特征选择，包括之前说的删除、合并一些特征

![image3431c.png](https://miao.su/images/2019/07/10/image3431c.png)

*如何解决？*
![bff740fb095ce62ddead5.png](https://miao.su/images/2019/07/10/bff740fb095ce62ddead5.png)

*在学习的时候，数据提供的特征有些影响模型复杂度或者这个特征的数据点异常较多，所以算法在学习的时候尽量减少这个特征的影响（甚至删除某个特征的影响），这就是正则化*

*注：调整时候，算法并不知道某个特征影响，而是去调整参数得出优化的结果*

#### 正则化类别

- L2正则化
  - 作用：可以使得其中一些W的都很小，都接近于0，削弱某个特征的影响
  - 优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象
  - Ridge回归
- L1正则化
  - 作用：可以使得其中一些W的值直接为0，删除这个特征的影响
  - LASSO回归

#### 拓展---原理

线性回归的损失函数用最小二乘法，等价于当预测值与真实值的误差满足正态分布时的极大似然估计；岭回归的损失函数，是最小二乘法+L2范数，等价于当预测值与真实值的误差满足正态分布，且权重值也满足正态分布（先验分布）时的最大后验估计；LASSO的损失函数，是最小二乘法+L1范数，等价于等价于当预测值与真实值的误差满足正态分布，且且权重值满足拉普拉斯分布（先验分布）时的最大后验估计。

* 解决过拟合：决策树---剪枝、线性回归---正则化
* 都有特征选择功能（嵌入式）



### 线性回归的改进---岭回归

#### 带有L2正则化的线性回归-岭回归

岭回归，其实也是一种线性回归。只不过在算法建立回归方程时候，加上正则化的限制，从而达到解决过拟合的效果

#### API

`sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True,solver="auto", normalize=False)`

- 具有l2正则化的线性回归
- alpha:正则化力度，也叫 λ
  - 正则化力度越大，权重系数会越小
  - 正则化力度越小，权重系数会越大
  - *λ取值：0~1 1~10*
- solver:会根据数据自动选择优化方法
  - *sag:如果数据集、特征都比较大，选择该随机梯度下降优化*
- normalize:数据是否进行标准化
  - normalize=False:可以在fit之前调用preprocessing.StandardScaler标准化数据
- Ridge.coef_:回归权重
- Ridge.intercept_:回归偏置

```pyhton
All last four solvers support both dense and sparse data. However,
only 'sag' supports sparse input when `fit_intercept` is True.
```

**该方法相当于SGDRegressor()**

sklearn.linear_model.RidgeCV(_BaseRidgeCV, RegressorMixin)

- 具有l2正则化的线性回归，可以进行交叉验证
- coef_:回归系数

```python
class _BaseRidgeCV(LinearModel):
    def __init__(self, alphas=(0.1, 1.0, 10.0),
                 fit_intercept=True, normalize=False, scoring=None,
                 cv=None, gcv_mode=None,
                 store_cv_values=False):
```

**观察正则化程度的变化，对结果的影响？**

![imagef9fe9.png](https://miao.su/images/2019/07/10/imagef9fe9.png)

- 正则化力度越大，权重系数会越小
- 正则化力度越小，权重系数会越大

#### 再论波士顿房价预测

```python
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
```



## 分类算法-逻辑回归与二分类

逻辑回归（Logistic Regression）是机器学习中的一种分类模型算法，虽然名字中带有回归，但是它只是与回归有一定的联系。由于算法的简单高效，在实际中应用非常广泛。

**应用场景**

- 广告点击率
- 是否为垃圾邮件
- 是否患病
- 金融诈骗
- 虚假账号

看到上面的例子，我们可以发现其中的特点，那就是都属于两个类别之间的判断。逻辑回归就是解决二分类问题的利器

### 原理

*输入：*
![7cf5494e2fa8093492849.png](https://miao.su/images/2019/07/11/7cf5494e2fa8093492849.png)

逻辑回归的输入就是一个线性回归的结果。

*激活函数：*

* sigmoid函数
  ![sigmoid4ae46.png](https://miao.su/images/2019/07/11/sigmoid4ae46.png)

* 分析
  - 回归的结果输入到sigmoid函数当中
  - 输出结果：[0, 1]区间中的一个概率值，默认为0.5为阈值

*逻辑回归最终的分类是通过属于某个类别的概率值来判断是否属于某个类别，并且这个类别默认标记为1(正例),另外的一个类别会标记为0(反例)。（方便损失计算）*

**输出结果解释(重要)：假设有两个类别A，B，并且假设我们的概率值为属于A(1)这个类别的概率值。现在有一个样本的输入到逻辑回归输出结果0.6，那么这个概率值超过0.5，意味着我们训练或者预测的结果就是A(1)类别。那么反之，如果得出结果为0.3那么，训练或者预测结果就为B(0)类别。**

*所以接下来我们回忆之前的线性回归预测结果我们用均方误差衡量，那如果对于逻辑回归，我们预测的结果不对该怎么去衡量这个损失呢？我们来看这样一张图*
![image9cd3d.png](https://miao.su/images/2019/07/11/image9cd3d.png)

那么如何去衡量逻辑回归的预测结果与真实结果的差异呢？（上图错了一个）

### 损失及优化

#### 损失

逻辑回归的损失称之为**对数似然损失**，公式如下：

* 分开类别：
  ![f1d6f0c0666393038b0b6.png](https://miao.su/images/2019/07/11/f1d6f0c0666393038b0b6.png)

怎么理解单个的式子呢？这个要根据log的函数图像来理解：
![imagee2fc7.png](https://miao.su/images/2019/07/11/imagee2fc7.png)

- 综合完整损失函数
  ![imageee26b.png](https://miao.su/images/2019/07/11/imageee26b.png)

> 看到这个式子，其实跟我们讲的信息熵类似。

接下来我们呢就带入上面那个例子来计算一遍，就能理解意义了。

![a502c467393f2b899ecf3.png](https://miao.su/images/2019/07/11/a502c467393f2b899ecf3.png)

我们已经知道，log(P), P值越大，结果越小，所以我们可以对着这个损失的式子去分析

#### 优化

同样使用*<u>梯度下降</u>*优化算法，去减少损失函数的值。这样去更新逻辑回归前面对应算法的权重参数， <u>*提升原本属于1类别的概率，降低原本是0类别的概率。*</u>

#### 拓展-关于逻辑回归的损失和线性回归的损失优化问题

均方误差这种损失函数，是一定能够通过梯度下降找到最优解。

### 逻辑回归API

sklearn.linear_model.LogisticRegression(solver='liblinear', penalty=‘l2’, C = 1.0)

- solver:优化求解方式（默认开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数）
  - sag：随机平均梯度下降
- penalty：正则化的种类
- C：正则化力度

*<u>默认将类别数量少的当做正例</u>*

### 案例

* 数据介绍
  ![ed1e5c3111a744e0d5fc6.png](https://miao.su/images/2019/07/11/ed1e5c3111a744e0d5fc6.png)
  原始数据的下载地址：[https://archive.ics.uci.edu/ml/machine-learning-databases/](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
* 数据描述
  1. 699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤相关的医学特征，最后一列表示肿瘤类型的数值。
  2. 包含16个缺失值，用”?”标出。

#### 分析

- 缺失值处理
- 标准化处理
- 逻辑回归预测

#### 代码

```python
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    return None


if __name__ == '__main__':
    logistic()
```

在很多分类场景当中我们不一定只关注预测的准确率！！！！！

比如以这个癌症举例子！！！我们并不关注预测的准确率，而是关注在所有的样本当中，癌症患者有没有被全部预测（检测）出来。

### 分类的评估方法

#### 精确率与召回率

##### 混淆矩阵

在分类任务下，预测结果(Predicted Condition)与正确标记(True Condition)之间存在四种不同的组合，构成混淆矩阵(适用于多分类)

![image2f8d8.png](https://miao.su/images/2019/07/11/image2f8d8.png)

##### 精确率（Precision）与召回率（Recall）

- 精确率：预测结果为正例样本中真实为正例的比例（了解）

![image9b924.png](https://miao.su/images/2019/07/11/image9b924.png)

- **召回率**：真实为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力）
- 召回率更加的重要

![imagee0da7.png](https://miao.su/images/2019/07/11/imagee0da7.png)

那么怎么更好理解这个两个概念：
![fa374c47bc496aa147fde.png](https://miao.su/images/2019/07/11/fa374c47bc496aa147fde.png)

还有其他的评估标准，F1-score，反映了模型的稳健性：
![F1cce16.png](https://miao.su/images/2019/07/11/F1cce16.png)

#### 分类评估报告API

- sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None )
- - y_true：真实目标值
  - y_pred：估计器预测目标值
  - labels:指定类别对应的数字
  - target_names：目标类别名称
  - return：每个类别精确率与召回率

```python
# 召回率
print("召回率为:", classification_report(y_test, log.predict(x_test), labels=[2, 4], target_names=['良性', '恶性']))
```

*假设这样一个情况，如果99个样本癌症，1个样本非癌症，不管怎样我全都预测正例(默认癌症为正例),准确率就为99%但是这样效果并不好，这就是样本不均衡下的评估问题*

*问题：如何衡量样本不均衡下的评估？*

### ROC曲线与AUC指标

#### 初识

- TPR = TP / (TP + FN)
  - 所有真实类别为1的样本中，预测类别为1的比例
- FPR = FP / (FP + FN)
  - 所有真实类别为0的样本中，预测类别为1的比例

#### ROC曲线

- ROC曲线的横轴就是FPRate，纵轴就是TPRate，当二者相等时，表示的意义则是：对于不论真实类别是1还是0的样本，分类器预测为1的概率是相等的，此时AUC为0.5

![ROC11dc6.png](https://miao.su/images/2019/07/11/ROC11dc6.png)

#### AUC指标

- AUC的概率意义是随机取一对正负样本，正样本得分大于负样本的概率
- AUC的最小值为0.5，最大值为1，取值越高越好
- <u>*AUC=1，完美分类器，采用这个预测模型时，不管设定什么阈值都能得出完美预测。绝大多数预测的场合，不存在完美分类器。*</u>
- <u>*0.5<AUC<1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。*</u>
- AUC=0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
- AUC<0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测，因此不存在 AUC<0.5 的情况。

*最终AUC的范围在[0.5, 1]之间，并且越接近1越好*

#### AUC计算API

from sklearn.metrics import roc_auc_score

- sklearn.metrics.roc_auc_score(y_true, y_score)
  - 计算ROC曲线面积，即AUC值
  - y_true:每个样本的真实类别，必须为0(反例),1(正例)标记
  - y_score:每个样本预测的概率值

```python
# 0.5~1之间，越接近于1约好
y_test = np.where(y_test > 2.5, 1, 0)

print("AUC指标：", roc_auc_score(y_test, log.predict(x_test)))
```

#### 总结

- AUC只能用来评价二分类
- AUC非常适合评价样本不平衡中的分类器性能
- AUC会比较预测出来的概率，而不仅仅是标签类



## 模型保存与加载

当训练或者计算好一个模型之后，那么如果别人需要我们提供结果预测，就需要保存模型（主要是保存算法的参数）

### sklearn模型的保存和加载API

from sklearn.externals import joblib

- 保存：joblib.dump(rf, 'test.pkl')
- 加载：estimator = joblib.load('test.pkl')

### 线性回归的模型保存和加载案例

* 保存

  ```python
  # 使用线性模型进行预测
  # 使用正规方程求解
  lr = LinearRegression()
  # 此时在干什么？
  lr.fit(x_train, y_train)
  # 保存训练完结束的模型
  joblib.dump(lr, "test.pkl")
  ```

* 加载

  ```python
  # 通过已有的模型去预测房价
  model = joblib.load("test.pkl")
  print("从文件加载进来的模型预测房价的结果：", std_y.inverse_transform(model.predict(x_test)))
  ```



## 无监督学习——K-means算法

### 什么是无监督学习

- 一家广告平台需要根据相似的人口学特征和购买习惯将美国人口分成不同的小组，以便广告客户可以通过有关联的广告接触到他们的目标客户。
- Airbnb 需要将自己的房屋清单分组成不同的社区，以便用户能更轻松地查阅这些清单。
- 一个数据科学团队需要降低一个大型数据集的维度的数量，以便简化建模和降低文件大小。

我们可以怎样最有用地对其进行归纳和分组？我们可以怎样以一种压缩格式有效地表征数据？<u>*这都是无监督学习的目标，之所以称之为无监督，是因为这是从无标签的数据开始学习的。*</u>

### 无监督学习相关算法

- 聚类
  - K-means(K均值聚类)
- 降维
  - PCA

### K-means原理

我们先来看一下一个K-means的聚类效果图：
![K-means62a97.png](https://miao.su/images/2019/07/11/K-means62a97.png)

### K-means聚类步骤

- 1、随机设置K个特征空间内的点作为初始的聚类中心（K值一般是确定的）
- 2、对于其他每个点计算到K个中心的距离，未知的点选择最近的一个聚类中心点作为标记类别
- 3、接着对着标记的聚类中心之后，重新计算出每个聚类的新中心点（平均值）
- 4、如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程

我们以一张图来解释效果：
![K-meanscdd58.png](https://miao.su/images/2019/07/11/K-meanscdd58.png)

### K-means API

sklearn.cluster.KMeans(n_clusters=8,init=‘k-means++’)

- k-means聚类
- n_clusters:开始的聚类中心数量
- init:初始化方法，默认为`k-means ++`
- labels_:默认标记的类型，可以和真实值比较（不是值比较）

### 案例

K-means对Instacart Market用户聚类

#### 分析

(待学习数据挖掘pandas等)

#### 代码

```
待更新
```

### K-means性能评估指标

#### 轮廓系数

![11fdcef1c3ca53a0bb8d0.png](https://miao.su/images/2019/07/11/11fdcef1c3ca53a0bb8d0.png)

*注：对于每个点i 为已聚类数据中的样本 ，b_i 为i 到其它族群的所有样本的距离最小值，a_i 为i 到本身簇的距离平均值。最终计算出所有的样本点的轮廓系数平均值*

#### 轮廓系数值分析

![0f1321333d3ece0155091.png](https://miao.su/images/2019/07/11/0f1321333d3ece0155091.png)

分析过程（我们以一个蓝1点为例）

1. 计算出$蓝_1$离本身族群所有点的距离的平均值$\overline a$
2. $蓝_1$到其它两个族群的距离计算出平均值红平均，绿平均，取最小的那个距离作为$\overline b$
3. 根据公式：极端值考虑：如果$\overline b$>>:$\overline a$ 那么公式结果趋近于1；如果$\overline a$>>$\overline b$>: 那么公式结果趋近于-1

#### 结论

如果$\overline b$>>$\overline a$:趋近于1效果越好， $\overline b$<<$\overline  a$:趋近于-1，效果不好。轮廓系数的值是介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优。

#### 轮廓系数API

sklearn.metrics.silhouette_score(X, labels)

- 计算所有样本的平均轮廓系数
- X：特征值
- labels：被聚类标记的目标值

#### 用户聚类结果评估

```python
silhouette_score(cust, pre)
```

### K-means总结

- 特点分析：采用迭代式算法，直观易懂并且非常实用
- 缺点：容易收敛到局部最优解(多次聚类)

*注意：聚类一般做在分类之前*