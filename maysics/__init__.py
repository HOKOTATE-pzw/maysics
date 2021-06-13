'''
本库用于科学计算和快速建模

maysics主要包括十二个模块：

1、algorithm 封装了几种模拟方法，用于简易模拟；
2、calculus 封装了部分常见的算符算子和积分方法，辅助数学运算；
3、constant 储存了部分常数；
4、equation 封装了部分方程求解运算；
5、explainer 用于评估和解释模型；
6、graph 用于图论分析；
7、models 封装了几种常用的模型以便快速构建数理模型；
8、preprocess 用于数据预处理；
9、stats 用于统计分析；
10、tprocess 用于处理时间数据；
11、transformation 储存了常用的坐标转换及其他数学变换；
12、utils 是额外工具箱。


This package is used for scientific calculating and fast modeling.

maysics includes twelve modules:

1. "algorithm" packages several simulation methods for simple simulation;
2. "calculus" packages some common operators and integration method to assist in mathematical operations;
3. "constant" contents some usual constants;
4. "equation" packages some equation solving operation;
5. "explainer" used for estimating and explaining model;
6. "graph" used for graph theory analysis;
7. "models" packages several commonly used models for fast modeling;
8. "preprocess" is used for data preproccessing;
9. "stats" is uesd for statistical analysis;
10. "tprocess" is used for processing time data;
11. "transformation" stores common coordinate transformations and other mathematical transformations;
12. "utils" is extra Utils.
'''
import numpy as np
import pickle, csv
from PIL import Image
from maysics import algorithm, calculus, constant, equation, explainer, graph,\
    models, preprocess, stats, tprocess, transformation, utils
from maysics.models import linear_r
from maysics.preprocess import preview, preview_file, shuffle
from maysics.utils import circle, discrete, grid_net


def covs1d(a, b, n):
    '''
    一维序列卷积和
    
    参数
    ----
    a：一维数组
    b：一维数组
    n：整型，平移步数
    
    返回
    ----
    数类型，a[n] * b[n]
    
    Convolution Sum of 1-D List
    
    Parameters
    ----------
    a: 1-D array
    b: 1-D array
    n: int, translation steps
    
    Return
    ------
    num, a[n] * b[n]
    '''
    a = np.array(a)
    b = list(b)
    b.reverse()
    b = np.array(b)
    num_a = len(a)
    num_b = len(b)
    if n <= 0 or n >= num_a + num_b:
        result = 0
    
    else:
        a = np.hstack((np.zeros(num_b), a, np.zeros(num_b)))
        b = np.hstack((b, np.zeros(num_a + num_b)))
        b[n : n+num_b] = b[: num_b]
        b[: n] = 0
        result = sum(a * b)
    
    return result


def covs2d(a, b, n, m):
    '''
    二维序列卷积和
    
    参数
    ----
    a：二维数组
    b：二维数组
    n：整型，沿axis=0方向的平移步数
    m：整型，沿axis=1方向的平移步数
    
    返回
    ----
    数类型，a[n, m] * b[n, m]
    
    Convolution Sum of 2-D List
    
    Parameters
    ----------
    a: 2-D array
    b: 2-D array
    n: int, translation steps along axis=0
    m: int, translation steps along axis=1
    
    Return
    ------
    num, a[n, m] * b[n, m]
    '''
    a = np.array(a)
    b = np.array(b)
    b = np.fliplr(b)
    b = np.flipud(b)
    num_a_x = a.shape[1]
    num_a_y = a.shape[0]
    num_b_x = b.shape[1]
    num_b_y = b.shape[0]
    
    if n <= 0 and m <= 0 or n >= num_a_x + num_b_x and m >= num_a_y + num_b_y:
        result = 0
    
    else:
        a = np.hstack((np.zeros((num_a_y, num_b_x)), a, np.zeros((num_a_y, num_b_x))))
        a = np.vstack((np.zeros((num_b_y, num_a_x + 2 * num_b_x)), a, np.zeros((num_b_y, num_a_x + 2 * num_b_x))))
        b = np.hstack((b, np.zeros((num_b_y, num_a_x + num_b_x))))
        b = np.vstack((b, np.zeros((num_a_y + num_b_y, num_a_x + 2 * num_b_x))))
        
        # 移动b矩阵
        b[n : n+num_b_y] = b[: num_b_y]
        b[: n] = 0
        b[:, m : m+num_b_x] = b[:, : num_b_x]
        b[:, : m] = 0
        
        result = (a * b).sum()
    
    return result


def save(filename, data, header=None):
    '''
    保存为.pkl、.npy或.csv文件
    
    参数
    ----
    filename：字符串类型，文件名
    data：需要保存的数据
    header：一维列表类型，可选，数据的列名称，仅在写入csv文件时有效
    
    
    Save as .pkl, .npy or .csv file
    
    Parameters
    ----------
    filename: str, file name
    data: data
    header: 1-D list, callable, the names of columns, effective only when writing csv files
    '''
    if filename[-4:] == '.pkl':
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    
    elif filename[-4:] == '.npy':
        np.save(filename, data)
    
    elif filename[-4:] == '.csv':
        data = np.array(data, dtype=np.object)
        if not header:
            header = []
            if len(data.shape) == 1:
                for i in range(data.shape[0]):
                    header.append(i)
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerow(data)
            
            else:
                for i in range(data.shape[1]):
                    header.append(i)
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(data)
    
    else:
        raise Exception("Suffix of filename must be '.pkl', '.npy' or '.csv'.")


def load(filename, header=True):
    '''
    载入.pkl、.npy或.csv文件
    
    参数
    ----
    filename：字符串类型，文件名
    header：布尔类型，可选，True表示csv文件第一行为列名，仅在读取csv文件时有效，默认为True
    
    
    Load .pkl, .npy or .csv file
    
    Parameter
    ---------
    filename: str, file name
    header: bool, callable, True means the first row of the csv file if the names of columns, effective only when reading csv files, default=True
    '''
    if filename[-4:] == '.pkl':
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        return data
    
    elif filename[-4:] == '.npy':
        return np.load(filename, allow_pickle=True)
    
    elif filename[-4:] == '.csv':
        with open(filename, 'r') as f:
            reader = list(csv.reader(f))
            if header:
                reader = reader[1:]
            return np.array(reader)
    
    else:
        raise Exception("Suffix of filename must be '.pkl', '.npy' or '.csv'.")


def pic_data(filename, dtype=np.uint8):
    '''
    将图片转换为张量数据
    
    参数
    ----
    filename：字符串类型，文件名
    dtype：可选，返回的元素类型，默认为np.uint8
    
    返回
    ----
    ndarray，图片的张量数据
    
    
    Transform the picture into tensor
    
    Parameters
    ----------
    filename: str, file name
    dtype: callable, the type of elements, default=np.uint8
    
    Return
    ------
    ndarray, the tensor of the picture
    '''
    x = Image.open(filename)
    return np.array(x, dtype=dtype)


def data_pic(data, filename):
    '''
    将张量数据转换为图片并保存
    
    参数
    ----
    data：列表类型，张量数据，列表的元素要求是整型
    filename：字符串类型，文件名
    
    
    Transform the tensor to the picture and save the picture
    
    Parameters
    ----------
    data: list, tensor, the elements of the data are required to be int
    filename: str, file name
    '''
    data = np.array(data, dtype=np.uint8)
    image = Image.fromarray(data)
    image.save(filename)


def all_same(x):
    '''
    判断数组元素是否全相同
    
    参数
    ----
    x：数组
    
    返回
    ----
    布尔类型，True或者False
    
    
    Determine whether the array elements are all the same
    
    Parameter
    ---------
    x: array
    
    Return
    ------
    bool, True or False
    '''
    x = np.array(x)
    if len(x.shape) == 1:
        x = len(set(x))
        if x == 1:
            return True
        else:
            return False
    
    else:
        for i in x:
            if i.all() != x[0].all():
                return False
        return True