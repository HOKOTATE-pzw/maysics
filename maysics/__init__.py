'''
本库用于科学计算和快速建模

maysics主要包括十一个模块：

1、algorithm 封装了几种模拟方法，用于简易模拟；
2、calculus 封装了部分常见的算符算子和积分方法，辅助数学运算；
3、constant 储存了部分常数；
4、equation 封装了部分方程求解运算；
5、graph 用于图论分析；
6、model_selection 用于评估和选择模型；
7、models 封装了几种常用的数学物理定律、方程、模型以便快速构建数理模型；
8、preprocess 用于数据预处理；
9、stats 用于统计分析；
10、transformation 储存了常用的坐标转换及其他数学变换；
11、utils 是额外工具箱。


This package is used for scientific calculating and fast modeling.

maysics includes eleven modules:

1. "algorithm" packages several simulation methods for simple simulation;
2. "calculus" packages some common operators and integration method to assist in mathematical operations;
3. "constant" contents some usual constants;
4. "equation" packages some equation solving operation;
5. "grapg" used for graph theory analysis;
6. "model_selection" used for estimating and selecting model;
7. "models" packages several commonly used laws, equations and models of mathematical physics for fast modeling;
8. "preprocess" is used for data preproccessing;
9. "stats" is uesd for statistical analysis;
10. "transformation" stores common coordinate transformations and other mathematical transformations;
11. "utils" is extra Utils.
'''

import numpy as np
import pickle
from maysics import algorithm, calculus, constant, equation, graph, model_selection, models, preprocess, stats, transformation, utils


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


def div(dividend, divisor):
    '''
    实现函数与同型函数、函数与数之间的除法
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


def save(data, filename):
    '''
    保存为.pkl或.npy文件
    
    参数
    ----
    data：需要保存的数据
    filename：字符串类型，文件名
    
    
    Save as .pkl or .npy file
    
    Parameters
    ----------
    data: data
    filename: str, file name
    '''
    if filename[-4:] == '.pkl':
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    
    elif filename[-4:] == '.npy':
        np.save(filename, data)
    
    else:
        raise Exception("Suffix of filename must be '.pkl' or '.npy'.")


def load(filename):
    '''
    载入.pkl或.npy文件
    
    参数
    ----
    filename：字符串类型，文件名
    
    
    Load .pkl or .npy file
    
    Parameter
    ---------
    filename: str, file name
    '''
    if filename[-4:] == '.pkl':
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        return data
    
    elif filename[-4:] == '.npy':
        return np.load(filename, allow_pickle=True)
    
    else:
        raise Exception("Suffix of filename must be '.pkl' or '.npy'.")