'''
本库用于科学计算和快速建模

maysics主要包括七个模块：

1、algorithm 储存了几种模拟方法，用于简易模拟；
2、constant 储存了部分常数；
3、models 储存了几种常用的数学物理定律、方程、模型以便快速构建数理模型；
4、model_selection 用于评估和选择模型；
5、operator 储存了部分常见的算符算子，辅助数学运算；
6、transformation 储存了常用的坐标转换及其他数学变换；
7、utils 是额外工具箱。


This package is used for scientific calculating and fast modeling.

maysics includes seven modules:

1. "algorithm" stores several simulation methods for simple simulation;
2. "constant" contents some usual constants;
3. "models" stores several commonly used laws, equations and models of mathematical physics for fast modeling;
4. "model_selection" used for estimating and selecting model;
5. "calculate" stores some common operators to assist in mathematical operations;
6. "transformation" stores common coordinate transformations and other mathematical transformations;
7. "utils" is extra Utils.

maysics-|- __init__
        |- algorithm -------|- GA
        |                   |- MC
        |                   |- SA
        |
        |- constant --------|- LP
        |
        |- model_selection -|- Error
        |                   |- Estimate
        |                   |- Search
        |                   |- Sense
        |
        |- models ----------|- ED
        |                   |- Fouriers_law
        |                   |- Leslie
        |                   |- Logistic
        |                   |- MVD_law
        |                   |- Plancks_law
        |
        |- operator --------|- Del
        |                   |- Dif
        |                   |- H
        |                   |- Laplace
        |
        |- transformation --|- Cylinder
        |                   |- Lorentz
        |                   |- Polar
        |                   |- Sphere
        |- utils
'''

import numpy as np


def arr(f):
    '''
    将矢量函数的输出形式统一为ndarray
    
    返回值：更改输出格式后的函数
    
    
    transform the output of vector function as ndarray
    
    return: function after changing the output format
    '''
    def obj(x):
        func = f(x)
        try:
            return np.array(func)
        except:
            return func
    return obj


def add(*arg):
    '''
    实现函数与同型函数、函数与数之间的加法
    要求作用函数若输出列表，必须是ndarray格式
    
    返回值：相加后的新函数
    
    
    addition between function and function or function and number
    if output of the function is list, it requires ndarray
    
    return: new function after addition
    '''
    def obj(x):
        list = []
        for i in range(len(arg)):
            if type(arg[i]).__name__ == 'function':
                list.append(arg[i](x))
            else:
                list.append(arg[i])
        return sum(list)
    return obj


def mul(*arg):
    '''
    实现函数与同型函数、函数与数之间的乘法
    要求作用函数若输出列表，必须是ndarray格式
    
    返回值：相乘后的新函数
    
    
    multiplication between function and function or function and number
    if output of the function is list, it requires ndarray
    
    return: new function after multiplication
    '''
    def obj(x):
        result = 1
        for i in range(len(arg)):
            if type(arg[i]).__name__ == 'function':
                result *= arg[i](x)
            else:
                result *= arg[i]
        return result
    return obj


def sub(minuend, subtrahend):
    '''
    实现函数与同型函数、函数与数之间的减法
    要求作用函数若输出列表，必须是ndarray格式
    minuend：被减数
    subtrahend：减数
    
    返回值：相乘后的新函数
    
    
    subtraction between function and function or function and number
    if output of the function is list, it requires ndarray
    minuend: minuend
    subtrahend: subtrahend
    
    return: new function after subtraction
    '''
    def obj(x):
        if type(minuend).__name__ == 'function':
            result_of_minuend = minuend(x)
        else:
            result_of_minuend = minuend
        if type(subtrahend).__name__ == 'function':
            result_of_subtrahend = subtrahend(x)
        else:
            result_of_subtrahend = subtrahend
        
        result = result_of_minuend - result_of_subtrahend
        return result
    return obj


def divi(dividend, divisor):
    '''
    实现函数与同型函数、函数与数之间的减法
    要求作用函数若输出列表，必须是ndarray格式
    
    参数
    ----
    dividend：被除数
    divisor：除数
    
    返回值：相乘后的新函数
    
    
    Parameters
    ----------
    division between function and function or function and number
    if output of the function is list, it requires ndarray
    dividend: dividend
    divisor: divisor
    
    return: new function after division
    '''
    def obj(x):
        if type(dividend).__name__ == 'function':
            result_of_dividend = dividend(x)
        else:
            result_of_dividend = dividend
        if type(divisor).__name__ == 'function':
            result_of_divisor = divisor(x)
        else:
            result_of_divisor = divisor
        result = result_of_dividend - result_of_divisor
        return result
    return obj


def r(*arg):
    '''
    相关系数
    
    返回值：各数组之间的相关系数矩阵
    
    
    correlation coefficient
    
    return: matrix of correlation coefficient
    '''
    arg = np.array(arg, dtype=float)
    if len(arg.shape) != 2:
        raise Exception("Input list should be 1 dimension.")
    
    cov_mat = np.cov(arg)
    var_mat = np.diagonal(cov_mat)**0.5
    
    for i in range(cov_mat.shape[0]):
        cov_mat[i] /= var_mat[i]
        cov_mat[:, i] /= var_mat[i]
    
    return cov_mat