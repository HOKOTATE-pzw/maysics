'''
本模块是额外工具箱

This module is extra Utils
'''
import numpy as np
from urllib import request
from lxml import etree
from urllib import parse
import string
from matplotlib import pyplot as plt
from maysics import trans


def grid_net(*args):
    '''
    生成网格点
    将输入的列表遍历组合
    如：
    grid_net([a, b], [c, d]) 或 grid_net(*[[a, b], [c, d]])
    返回：array([[a, c], [b, c], [a, d], [b, d]])
    
    
    Generate grid
    traverse and combine the input list
    e.g.
    grid_net([a, b], [c, d]) or grid_net(*[[a, b], [c, d]])
    return: array([[a, c], [b, c], [a, d], [b, d]])
    '''
    net = np.meshgrid(*args)
    for i in range(len(net)):
        net[i] = net[i].flatten()
    net = np.vstack(tuple(net)).T
    return net


def group(data, index, f, param={}):
    '''
    分组处理数据
    
    参数
    ----
    data：二维ndarray，数据
    index：整型，需要分组处理的列索引
    f：函数类型，将分组后的每个部分的二维ndarray作为输入
    param：字典类型，可选，用于传递f中的其他参数，默认为空字典
    
    返回
    ----
    二维ndaray
    
    
    Group process data
    
    Paramters
    ---------
    data: 2D ndarray, data
    index: int, column to be grouped
    f: function, take 2D ndarray of each part after grouping as input
    param: dict, callable, pass other parameters to f, default={}
    
    Return
    ------
    2D ndarray
    '''
    data_index = data[:, index]
    set_ls = set(data_index)
    for i in set_ls:
        data_new = data[data_index==i, :]
        data_new = f(data_new, **param)
        data[data_index==i, :] = data_new
    return data


def hermit(data):
    '''
    求厄密共轭矩阵
    
    参数
    ----
    data：二维数组
    
    返回
    ----
    二维ndarray
    
    
    Hermitian Conjugate Matrix
    
    Parameter
    ---------
    data: 2-D array
    
    Return
    ------
    2-D ndarray
    '''
    data = np.array(data)
    return data.T.conj()


def mat_exp(x, tol=0.01):
    '''
    矩阵的exp运算
    求e^data
    
    参数
    ----
    x：二维数组，矩阵
    tol：浮点数类型，可选，误差，当data^n/n!的每个矩阵元均小于tol时输出结果
    
    返回
    ----
    二维ndarray
    
    
    Exp Operation of Matrix
    
    Parameters
    ----------
    x: 2-D array, matrix
    tol: float, callable, error, output when every element of data^n/n! less than tol
    
    Return
    ------
    2-D ndarray
    '''
    x = np.matrix(x)
    I_x = np.matrix(np.eye(x.shape[0]))
    result_up = I_x.copy()
    error = I_x.copy()
    result = I_x.copy()
    n = 0
    result_down = 1
    while (error > tol).any():
        n += 1
        result_up *= x
        result_down *= n
        error = result_up / result_down
        result += error
    return np.array(result)


def e_distance(p1, p2):
    '''
    求某两个点之间的欧式距离
    
    参数
    ----
    p1：一维数组，第一个点的位置
    p2：一维数组，第二个点的位置
    
    返回
    ---
    浮点数类型，距离
    
    
    Calculate the Euclidean distance between two points
    
    Parameters
    ----------
    p1: 1-D array, the location of the first point
    p2: 1-D array, the location of the second point
    
    Return
    ------
    float, the distance
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return sum((p1 - p2)**2)**0.5


def e_distances(data, des='o'):
    '''
    求data到目标点的欧式距离
    
    参数
    ----
    data：一维或二维数组，数据
    des：字符串或一维数组，可选'o'或'O'(原点)、'mean'(均值点)及自定义数组，目标点坐标，默认为'o'
    
    返回
    ----
    ndarray类型，距离数组
    
    
    Calculate the Euclidean distances between data and destination
    
    Parameter
    ---------
    data: 1-D or 2-D array, data
    des: str or 1-D array, 'o' or 'O' (origin), 'mean' (mean point) and custom array are optional, the coordinate of destination, default='o'
    
    Return
    ------
    ndarray, the distances
    '''
    data = np.array(data, dtype=np.float)
    if len(data.shape) < 3:
        n = data.shape[-1]
    else:
        raise Exception("Parameter 'data' must be 1-D or 2-D.")

    if des == 'o' or des == 'O':
        des = np.zeros(n)

    elif des == 'mean':
        des = data.mean(axis=0)
        
    else:
        des = np.array(des)
    
    data = (data - des)**2
    result = data.sum(axis=len(data.shape)-1)
    result = result**0.5
    
    return result


def m_distance(data, p1, p2):
    '''
    求某两个点之间的马氏距离
    
    参数
    ----
    data：二维列表，数据
    p1：一维或二维数组，第一个点的位置
    p2：一维或二维数组，第二个点的位置
    
    返回
    ---
    浮点数类型，距离
    
    
    Calculate the Mahalanobis distance between two points
    
    Parameters
    ----------
    data: 2-D list, data
    p1: 1-D or 2-D list, the location of the first point
    p2: 1-D or 2-D list, the location of the second point
    
    Return
    ------
    float, the distance
    '''
    data = np.mat(data, dtype=np.float)
    dataT = data.T
    
    if len(data.shape) != 2:
        raise Exception("Parameter 'data' must be 2-D.")
    
    SI = np.mat(np.cov(dataT)).I

    p1 = np.mat(p1, dtype=np.float)
    p2 = np.mat(p2, dtype=np.float)
    result = (p1 - p2) * SI * (p1 - p2).T
    
    return result[0, 0]**0.5


def m_distances(data, des='o'):
    '''
    求data到目标点的马氏距离
    
    参数
    ----
    data：二维数组，数据
    des：字符串或一维或二维数组，可选'o'或'O'(原点)、'mean'(均值点)及自定义数组，目标点坐标，默认为'o'
    
    返回
    ----
    ndarray类型，距离数组
    
    
    Calculate the Mahalanobis distance between data and destination
    
    Parameter
    ---------
    data: 2-D array, data
    des: str or 1-D or 2-D array, 'o' or 'O' (origin), 'mean' (mean point) and custom array are optional, the coordinate of destination, default='o'
    
    Return
    ------
    ndarray, the distances
    '''
    data = np.mat(data, dtype=np.float)
    dataT = data.T
    
    if len(data.shape) != 2:
        raise Exception("Parameter 'data' must be 2-D.")
    
    SI = np.mat(np.cov(dataT)).I

    if des == 'o' or des == 'O':
        des = np.zeros(data.shape[-1])
    
    elif des == 'mean':
        des = data.mean(axis=0)
    
    else:
        des = np.mat(des)
    
    return np.diag((data - des) * SI *(dataT - des.T))**0.5


def discrete(x, y, color=None, label=None):
    '''
    绘制离散函数图像
    
    参数
    ----
    x：一维数组，自变量
    y：一维数组，因变量
    color：字符串类型，可选，颜色
    label：字符串类型，可选，标签
    
    
    Draw the graph of discrete function
    
    Parameters
    ----------
    x: 1-D array, independent variable
    y: 1-D array, dependent variable
    color: str, callable, color
    label: str, callable, label
    '''
    plt.scatter(x, y, color=color, label=label)
    zeros = np.zeros_like(y)
    if not color:
        plt.vlines(x, zeros, y)
    else:
        plt.vlines(x, zeros, y, color=color)


def circle(center=(0, 0), radius=1, angle_range=(0, 2*np.pi), acc=0.01, c=None, label=None):
    '''
    绘制一个圆
    
    参数
    ----
    center：元组类型，可选，圆心坐标，默认为(0, 0)
    radius：数类型，可选，半径，默认为1
    angle_range：元组类型，可选，绘制的角度范围，默认为(0, 2π)
    acc：浮点数类型，可选，绘制的精度，默认为0.01
    c：字符串类型，可选，颜色
    label：字符串类型，可选，标签，默认为None
    
    
    Draw a circle
    
    Parameters
    ----------
    center: tuple, callable, center coordinate, default=(0, 0)
    radius: num, callable, radius, default=1
    angle_range: tuple, callable, the range of angle to draw, default=(0, 2π)
    acc: float, callable, the accuracy of drawing, default=0.01
    c: str, callable, color
    label: str, callable, label, default=None
    '''
    theta = np.arange(*angle_range, acc)
    radius = radius * np.ones_like(theta)
    x = np.vstack((radius, theta)).T
    x = trans.ipolar(x)
    plt.plot(x[:, 0] + center[0], x[:, 1] + center[1], c=c)


class A_P():
    '''
    将信号的频域表示分解为“幅度-频率”和“相位-频率”

    参数
    ----
    X：函数或一维数组形式，信号的频域表示


    Decompose frequency domain representation of signal into "amplitude-frequency" and "phase-frequency"

    Parameter
    ---------
    X: function or 1-D array, frequency domain representation of signal
    '''
    def __init__(self, X):
        self.X = X
    

    def fit(self, f, param={}):
        '''
        计算频率为f时的幅度和相位

        参数
        ----
        f：函数或一维数组形式，频率
        param：字典类型，可选，用于传递f中的其他参数，仅当f为函数类型时有效，默认为空字典

        返回
        ----
        元组形式，(幅度, 相位)


        Calculate the amplitude and phase at frequency f

        Parameter
        ---------
        f: function or 1-D array, frequency
        param: dict, callable, pass other parameters to f, valid only when f is a function, default={}

        Return
        ------
        tuple, (amplitude, phase)
        '''
        self.f = np.array(f, np.float)
        f = self.f.astype(complex)
        if type(self.X).__name__ == 'function':
            result = self.X(f, **param)
        else:
            X = np.array(self.X)
            result = self.X[f]
        result = np.array(result)
        self.amplitude = abs(result)
        
        index1 = np.where(result.imag == 0)[0]
        index2 = np.where(result.imag != 0)[0]
        result_new = result[index1]
        result_new[result_new.real == 0] = 0
        result_new[result_new.real > 0] = np.pi / 2
        result_new[result_new.real < 0] = -np.pi / 2
        result[index1] = result_new
        result_new = result[index2]
        result[index2] = np.arctan(result_new.imag / result_new.real)
        self.phase = result.real
    

    def __image_process(self, image_type):
        fig = plt.figure()
        if image_type == 'C' or image_type == 'c':
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(self.f, self.amplitude)
            ax.set_title('amplitude')
            ax = fig.add_subplot(2, 1, 2)
            ax.plot(self.f, self.phase)
            ax.set_title('phase')
        elif image_type == 'D' or image_type == 'd':
            zeros_list = np.zeros(self.f.shape)
            ax = fig.add_subplot(2, 1, 1)
            ax.scatter(self.f, self.amplitude, marker='o', s=30, zorder=3)
            ax.vlines(self.f, zeros_list, self.amplitude)
            ax.set_title('amplitude')
            ax = fig.add_subplot(2, 1, 2)
            ax.scatter(self.f, self.phase, marker='o', s=30, zorder=3)
            ax.vlines(self.f, zeros_list, self.phase)
            ax.set_title('phase')
        plt.tight_layout()


    def show(self, image_type='c'):
        '''
        显示“幅度-频率”图和“相位-频率”图

        参数
        ----
        image_type：字符串形式，可选'c'和'd'，'c'表示绘制连续图像，'d'表示绘制离散图像，默认为'd'


        Display "amplitude-frequency" and "phase-frequency" graphs

        Parameter
        ---------
        image_type: str, 'c' and 'd' are callable, 'c' means drawing continuous image and 'd'means drawing discrete image, default='c'
        '''
        self.__image_process(image_type)
        plt.show()
    

    def savefig(self, filename, image_type='c'):
        '''
        储存“幅度-频率”图和“相位-频率”图

        参数
        ----
        filename：字符串形式，文件名
        image_type：字符串形式，可选'C'和'D'，'C'表示绘制连续图像，'D'表示绘制离散图像，默认为'C'


        Save "amplitude-frequency" and "phase-frequency" graphs

        Parameters
        ----------
        filename: str, file name
        image_type: str, 'c' and 'd' are callable, 'c' means drawing continuous image and 'd'means drawing discrete image, default='c'
        '''
        self.__image_process(image_type)
        plt.savefig(filename)