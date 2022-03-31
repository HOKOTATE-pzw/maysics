'''
本库用于科学计算和快速建模

maysics主要包括十二个模块：

1、algo 封装了几种模拟方法，用于简易模拟；
2、calc 封装了部分常见的算符算子和积分方法，辅助数学运算；
3、equa 封装了部分方程求解运算；
4、explain 用于评估和解释模型；
5、graph 用于图论分析；
6、image 用于数字图像处理;
7、models 封装了几种常用的模型以便快速构建数理模型；
8、prep 用于数据预处理；
9、stats 用于统计分析；
10、time 用于处理时间数据；
11、trans 储存了常用的坐标转换及其他数学变换；
12、utils 是额外工具箱。


This package is used for scientific calculating and fast modeling.

maysics includes twelve modules:

1. "algo" packages several simulation methods for simple simulation;
2. "calc" packages some common operators and integration method to assist in mathematical operations;
3. "equa" packages some equation solving operation;
4. "explain" is used for estimating and explaining model;
5. "graph" is used for graph theory analysis;
6. "image" is used for digital image process;
7. "models" packages several commonly used models for fast modeling;
8. "prep" is used for data preproccessing;
9. "stats" is uesd for statistical analysis;
10. "time" is used for processing time data;
11. "trans" stores common coordinate transformations and other mathematical transformations;
12. "utils" is extra Utils.
'''
import numpy as np
from scipy.special import factorial
import pickle, csv
from PIL import Image
import random
from . import algo, calc, equa, explain, graph, image,\
    models, prep, stats, time, trans, utils
from .models import linear_r
from .prep import preview, preview_file, shuffle, standard, minmax, normalizer
from .utils import circle, discrete, grid_net, hermit, mat_exp


# math
pi = 3.141592653589793
e = 2.718281828459045
golden = 1.618033988749895
gamma = 0.57721566490153286060651209
K = 0.76422365358922066299069873125
chaos_1 = 4.669201609102990
chaos_2 = 2.502907875095892
K0 = 2.6854520010


# physics
atm = 101325
c = 299792458.0
c_e = 1.602176634e-19
m_e = 9.1093837015e-31
m_earth = 5.965e24
m_n = 1.67492749804e-27
m_p = 1.67262192369e-27
m_sun = 1.9891e30
G = 6.6743e-11
g = 9.80665
h = 6.62607015e-34
hr = 1.0545718176461565e-34
k = 1.380649e-23
miu = 1.2566370614359173e-06
epsilon = 8.85e-12
NA = 6.02214179e23
R = 8.31446114959365
r_earth = 6371393
r_sun = 6.955e8
r_moon = 384403000
lambdac = 2.4263102175e-12
ly = 9460730472580800
AU = 149597870700
pc = 3.0856775814671915808e16
Platonic_year = 26000
SB = 5.670367e-8
v1 = 7900
v2 = 11200
v3 = 16700


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


def choice(seq, pro=None, random_state=None):
    '''
    按指定概率抽取元素
    
    参数
    ----
    seq：一维列表，待抽取的元素
    pro：一维数组，可选，抽取相应元素的概率，默认概率全部相等
    random_state：整型，可选，随机种子
    
    返回
    ----
    seq中的元素
    
    
    Chooce elements according to the specified probability
    
    Parameters
    ----------
    seq: 1-D array, elements to be chooced
    pro: 1-D array, callable, the probability of choocing each element, the probability is equal by default
    random_state: int, callable, random seed
    
    Return
    ------
    the element in seq
    '''
    if not random_state is None:
        random.seed(random_state)
    
    if pro is None:
        return random.choice(seq)
    else:
        num_pro = len(pro)
        for i in range(1, num_pro):
            pro[i] += pro[i-1]
        random_num = random.random()
        for i in range(num_pro):
            if pro[i] >= random_num:
                return seq[i]


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


def load(filename, header=True, pic=False, dtype='uint8'):
    '''
    载入pkl、npy、csv文件或图片
    
    参数
    ----
    filename：字符串类型，文件名
    header：布尔类型，可选，True表示csv文件第一行为列名，仅在读取csv文件时有效，默认为True
    pic：布尔类型，可选，True表示读取图片，默认为False
    dtype：可选，输出图像数据类型，仅在pic=True时有效，默认为'uint8'
    
    
    Load pkl, npy, csv file or picture
    
    Parameter
    ---------
    filename: str, file name
    header: bool, callable, True means the first row of the csv file if the names of columns, effective only when reading csv files, default=True
    pic: bool, callable, True means to load picture, default=False
    dtype: callable, data format of output image, effective only when pic=True, default='uint8'
    '''
    if pic is False:
        if filename[-4:] == '.pkl':
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            
            return data
        
        elif filename[-4:] == '.npy':
            return np.load(filename, allow_pickle=True)
        
        elif filename[-4:] == '.csv':
            with open(filename, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
                if header:
                    reader = reader[1:]
                return np.array(reader)
        
        else:
            raise Exception("Suffix of filename must be '.pkl', '.npy' or '.csv'.")
    
    else:
        x = Image.open(filename)
        return np.asarray(x, dtype=dtype)


def save(filename, data, header=None, pic=False):
    '''
    保存为pkl、npy、csv文件或图片
    
    参数
    ----
    filename：字符串类型，文件名
    data：需要保存的数据
    header：一维列表类型，可选，数据的列名称，仅在写入csv文件时有效
    pic：布尔类型，可选，True表示保存为图片，默认为False
    
    
    Save as pkl, npy, csv file or picture
    
    Parameters
    ----------
    filename: str, file name
    data: data
    header: 1-D list, callable, the names of columns, effective only when writing csv files
    pic: bool, callable, True means to save as picture, default=False
    '''
    if pic is False:
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
    
    else:
        data = np.array(data, dtype='uint8')
        image = Image.fromarray(data)
        image.save(filename)


def lpn(l):
    '''
    l阶勒让德多项式的模
    
    参数
    ----
    l：整型，阶数
    
    返回
    ----
    数类型
    
    
    Module of Legendre Polynomials at degree l
    
    Paramater
    ---------
    l: int, degree
    
    Return
    ------
    num
    '''
    return (2 / (2 * l + 1))**0.5


def lp(x, l):
    '''
    l阶勒让德多项式的值
    
    参数
    ----
    x：数，输入值
    l：整型，阶数
    
    返回
    ----
    数
    
    
    Value of Legendre Polynomials at degree l
    
    Parameters
    ----------
    x: num, input value
    l: int, degree
    
    Return
    ------
    num
    '''
    result = 0
    for k in range(int(l/2) + 1):
        result += (-1)**k * factorial(2 * l - 2 * k)\
        / 2**l / factorial(k) / factorial(l - k) / factorial(l - 2 * k)\
        * x**(l - 2 * k)
    return result


def alpn(l, m):
    '''
    m阶l次连带勒让德多项式的模
    
    参数
    ----
    l：整型，次数
    m：整型，阶数
    
    Value of Associated Legendre Polynomials of order m and degree l
    
    Parameters
    ----------
    l: int, degree
    m: int, order
    '''
    up_part = 2 * factorial(l + m)
    down_part = (2 * l + 1) * factorial(l - m)
    return (up_part / down_part)**0.5


def _dm_lp(x, l, m, acc):
    '''
    勒让德多项式的m阶导数
    '''
    if m == 0:
        x = lp(x, l)
    elif m == 1:
        x = (lp(x + acc * 0.5, l) - lp(x - acc * 0.5, l)) / acc
    else:
        x = (_dm_lp(x + acc * 0.5, l, m-1, acc) - _dm_lp(x - acc * 0.5, l, m-1, acc)) / acc
    return x


def alp(x, l, m, acc=0.1):
    '''
    m阶l次连带勒让德多项式的值
    Plm(x) = (1-x^2)^(m/2) * d^m/dx^m * Pl(x)
    
    参数
    ----
    x：数，输入值
    l：整型，次数
    m：整型，阶数
    acc：浮点数类型，求导精度
    
    
    Value of Associated Legendre Polynomials of order m and degree l
    Plm(x) = (1-x^2)^(m/2) * d^m/dx^m * Pl(x)
    
    Parameters
    ----------
    x: num, input value
    l: int, degree
    m: int, order
    acc: float, callable, accuracy of derivation, default=0.1
    '''
    return (1 - x**2)**(m / 2) * _dm_lp(x, l, m, acc)


def hp(x, n, li=False):
    '''
    n阶厄米多项式

    参数
    ----
    x：数，输入值
    n：整型，阶数
    li：布尔类型，可选，True表示将0到n阶勒让德多项式组成一维ndarray返回，默认为False

    返回
    ----
    数或一维ndarray


    Value of Hermite Polynomials at degree n

    Parameters
    ----------
    x: num, input value
    n: int, degree
    li: bool, callable, True means that 0 to n order Legendre polynomials are formed into 1-D ndarray and returned, default=False
    
    Reuturn
    -------
    num or 1-D ndarray
    '''
    if n <= 0:
        n = 0
    
    if not li:
        if n == 0:
            return 1

        elif n == 1:
            return 2 * x

        else:
            return 2 * x * hp(x, n-1) - 2 * (n-1) * hp(x, n-2)
    
    else:
        hp_list = []
        for i in range(n+1):
            if i == 0:
                hp_list.append(1)
            
            elif i == 1:
                hp_list.append(2 * x)
            
            else:
                hp_list.append(2 * x * hp_list[i-1] - 2 * (i-1) * hp_list[i-2])
        return np.array(hp_list)


def v_mean(m, T):
    '''
    麦克斯韦速率分布律下的平均速率
    
    参数
    ----
    m：气体分子质量, 单位：kg
    T：气体温度, 单位：K
    
    返回
    ----
    数或数组类型，速率，单位：m/s
    
    
    Average Velocity in Maxwell's Velocity Distribution Law
    
    Parameters
    ----------
    m: mass of gas molecule, unit: kg
    T: temperature of gas, unit: K
    
    Return
    ------
    num or array, velocity, unit: m/s
    '''
    return 8**0.5 * (k * T / (np.pi * m))**0.5


def v_p(m, T):
    '''
    麦克斯韦速率分布律下的最概然速率
    
    参数
    ----
    m：气体分子质量, 单位：kg
    T：气体温度, 单位：K
    
    返回
    ----
    数或数组类型，速率，单位：m/s
    
    
    Most Probable Velocity in Maxwell's Velocity Distribution Law
    
    Parameters
    ----------
    m: mass of gas molecule, unit: kg
    T: temperature of gas, unit: K
    
    Return
    ------
    num or array, velocity, unit: m/s
    '''
    return 2**0.5 * (k * T / (np.pi * m))**0.5


def v_rms(m, T):
    '''
    麦克斯韦速率分布律下的均方根速率
    
    参数
    ----
    m：气体分子质量, 单位：kg
    T：气体温度, 单位：K
    
    返回
    ----
    数或数组类型，速率，单位：m/s
    
    
    Root-Mean-Square Velocity Maxwell's Velocity Distribution Law
    
    Parameters
    ----------
    m: mass of gas molecule, unit: kg
    T: temperature of gas, unit: K
    
    Return
    ------
    num or array, velocity, unit: m/s
    '''
    return 3**0.5 * (k * T / (np.pi * m))**0.5