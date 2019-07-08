# 人工智能の数据挖掘

[TOC]

## 数据挖掘基础环境安装与使用

* 完成数据挖掘基础阶段的所有环境安装
* 应用 `jupyter notebook` 完成代码编写运行

### 库的安装

整个数据挖掘基础阶段会用到`Matplotlib、Numpy、Pandas、Ta-Lib`等库，为了统一版本号在环境中使用，将所有的库及其版本放到了文件`requirements.txt`中，然后统一安装

*新建一个用于人工智能环境的虚拟环境*

```shell
mkvirtualenv -p python3 ai
```

```shell
matplotlib==2.0.2
numpy==1.14.2
pandas==0.20.3
TA-Lib==0.4.16
tables==3.4.2
jupyter==1.0.0
```

使用pip命令安装：

```shell
pip install -r requirements.txt
```

## Jupyter notebook使用

### 介绍

Jupyter项目是一个非盈利的开源项目，源于2014年的ipython项目，因为它主键发展为支持跨所有编程语言的交互式数据科学和科学计算

* Jupyter Notebook，原名Ipython Notebook，是Ipython的加强网页版，一个开源Web应用程序
* 名字源自Julia、Python和R（数据科学的三种开源语言）
* **.ipynb**文件格式是用于计算型叙述的**JSON文档格式**的正式规范
  ![image51c1e.png](https://miao.su/images/2019/06/23/image51c1e.png)

### 为什么要使用Jupyter

* 传统软件开发：工程/目标明确
  * 需求分析、设计架构、开发模块、测试
* 数据挖掘：艺术/目标不明确
  * 目的是具体的洞察目标，而不是机械的完成任务
  * 通过执行代码来理解问题
  * 迭代式地改进代码来改进解决方法

实时运行的代码、叙事性的文本和可视化被整合在一起，方便使用代码和数据来讲述故事。

### Jupyter使用

界面启动、新建

**本机打开**

```shell
# 进入虚拟环境
workon ai
# 输入命令
jupyter notebook
```

* 本地notebook的默认URL为：`http://localhost:8888`
* 需要配置 `~/.jupyter/jupyter notebook config`
* 想让notebook打开指定目录，只要进入此目录后执行命令即可

![image0ab39.png](https://miao.su/images/2019/06/24/image0ab39.png)

新建notebook文档：
![image2f233.png](https://miao.su/images/2019/06/24/image2f233.png)

内容界面：
![image75cf8.png](https://miao.su/images/2019/06/24/image75cf8.png)

- 标题栏：点击标题（如Untitled）修改文档名
- 菜单栏
  - 导航-File-Download as，另存为其他格式
  - 导航-Kernel
    - Interrupt，中断代码执行（程序卡死时）
    - Restart，重启Python内核（执行太慢时重置全部资源）
    - Restart & Clear Output，重启并清除所有输出
    - Restart & Run All，重启并重新运行所有代码

### cell

#### cell操作

cell：一堆In Out会话被视作一个单元代码，称为cell
Jupyter支持两种模式：

* 编辑模式（Enter）
  * 命令模式下`回车Enter`或`鼠标双击`cell进入编辑模式
  * 可以**操作cell内文本**或代码，剪切/复制/粘贴移动等操作
* 命令模式（Esc）
  * 按`Esc`退出编辑，进入命令模式
  * 可以**操作cell单元本身**进行剪切/复制/粘贴/移动等操作

![imagefe989.png](https://miao.su/images/2019/06/25/imagefe989.png)

#### 快捷键操作cell

* 两种模式通用跨快捷键
  * `Shift+Enter`，执行本单元代码，并跳转下一单元
  * `Ctrl+Enter`，执行本单元代码，留在本单元
    cell行号签的`*`，表示代码正在运行
* 命令模式：按`Esc`进入
  * `Y`，cell切换到Code模式
  * `M`，cell切换到Markdown模式
  * `A`，在当前cell的上面添加cell
  * `B`，在当前cell的下面添加cell
  * `双击D`，删除当前cell
  * `Z`，回退
  * `L`，为当前cell加上行号
  * `Ctrl+Shift+P`，对话框输入命令直接运行
  * 快速跳转到首个cell，`Ctrl+Home`
  * 快速跳转到最后一个cell,`Ctrl+End`
* 编辑模式：按`Enter`进入
  * 多光标操作：`Ctrl键点击鼠标`
  * 回退：`Ctrl+Z`
  * 重做：`Ctrl+Y`
  * 补全代码：变量、方法后跟`Tab键`
  * 为一行或多行代码添加注释：`Ctrl+/`
  * 屏蔽自动输出信息：可在最后一条语句之后加一个分号

## Matplotlib

### 绘图架构

#### 什么是Matplotlib

![imageb3aad.png](https://miao.su/images/2019/06/25/imageb3aad.png)

* 是专门用于开发2D图表（包括3D图表）
* 使用起来及其简单
* 以渐进、交互式方式实现数据可视化

#### 为什么要学习Matplotlib

可视化是整个数据挖掘中的关键辅助工具，可以清晰的理解数据，从而调整我们的分析方法。

* 能将数据进行可视化，更直观的呈现
* 使数据更加客观、更具说服力

例如下面两个图为数字展示和图形展示：
![imaged9e44.png](https://miao.su/images/2019/06/25/imaged9e44.png)

我们先来简单画一个图看一下效果：
![image4bca8.png](https://miao.su/images/2019/06/25/image4bca8.png)

### Matplotlib框架结构

matplotlib框架分为三层，这三层构成了一个栈，上层可以调用下层。
![image40eea.png](https://miao.su/images/2019/06/25/image40eea.png)

#### 后端层

matplotlib的底层，实现了大量的抽象接口类，这些API用来在底层实现图形元素的一个个类。

* FigureCanvas对象 实现了绘图区域这一概念
* Renderer对象 在FigureCanvas上绘图

#### 美工层

图形中所有能看到的元素都属于Artist对象，即标题、轴标签、刻度等组成图形的所有元素都是Artist对象的实例：

* Figure：指整个图形（包括所有的元素，比如标题、线等）
* Axes（**坐标系**）：数据的绘图区域
* Axis（**坐标轴**）：坐标系中的一条轴，包含大小限制、刻度和刻度标签

特点为：

* 一个figure（图）可以包含多个axes（坐标系），但是一个axes只能属于一个figure
* 一个axes（坐标系）可以包含多个axis（坐标轴），包含两个即为2d坐标系

![imagedc781.png](https://miao.su/images/2019/06/25/imagedc781.png)

#### 脚本层

主要用于可视化编程，pyplot模块可以提供给我们一个与matplotlib打交道的接口。可以只通过调用pyplot模块的函数从而操作整个程序包，来绘制图形。

* 操作或者改动Figure对象，例如创建Figure对象
* 大部分工作是处理样本文件的图形与坐标的生成

## 折线图与基础绘图功能

Parts of  a Figure
![imagedfe6d.png](https://miao.su/images/2019/06/25/imagedfe6d.png)

### 折线图绘制与保存图片

为了更好的去理解所有基础绘图功能，我们通过天气温度变化的绘图来融合所有的基础API使用。

#### matplotlib.pyplot模块

matplotlib.pyplot包含了一系列类似于matlab的画图函数。它的函数*作用于当前图形（figure）的当前坐标系（axes）*。

```python
# 为方便简介,简称为plt
import matplotlib.pyplot as plt
```

#### 折线图绘制与显示

*展现上海一周的天气，比如从星期一到星期日的天气温度如下*：
![image4a32d.png](https://miao.su/images/2019/06/25/image4a32d.png)

可以看到这样去显示效果并不好，图形的大小等等，所以我们可以通过加入更多的功能。

#### 修改图形大小与图片保存

```python
plt.figure(figsize=(x, x), dpi = x)
	figsize：指定图的长和宽
	dpi：指定图像的清晰度
	返回fig对象
plt.savefig(path)
```

![image31b2c.png](https://miao.su/images/2019/06/25/image31b2c.png)

### 温度变化表示

需求：画出某城市11点到12点1小时内每分钟的温度变化折线图，温度范围在15度到18度

效果如下：

![image853f9.png](https://miao.su/images/2019/06/25/image853f9.png)

#### 构造数据、显示

![imagedc4fd.png](https://miao.su/images/2019/06/26/imagedc4fd.png)

#### 自定义 $x$轴和 $y$轴刻度以及中文显示

* plt.xticks(x, **kwargs)
  x：要显示的刻度值

* plt.yticks(y, **kwargs)
  y：要显示的刻度值

  

  