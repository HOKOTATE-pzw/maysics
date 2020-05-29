'''
本模块储存着常用算符
调用模块中的算符时应注意：
1、算符作用的多元函数的自变量需为列表(list、tuple、ndarray均可，下同)
2、算符作用的标量函数的输出值需为单个值
3、算符作用的矢量函数的输出值需为列表

This module stores the common operators
When calling operators in the module, pay attention to:
1. The argument of the multivariate function acted by the operators should be list (list, tuple, ndarray, the same below)
2. The output value of scalar function acted by the operators should be a single value
3. The output value of vector function acted by the operators should be list
'''

import numpy as np
from scipy import constants as C


class Dif():
    '''
    对多元函数的求导算子，对矢量函数则相当于对每个分量函数都进行了相同的求导
    多元矢量函数可以分解为多个标量函数
    无法将二维列表识别成一行作为一个自变量，需要循环
    
    参数
    ----
    dim：整型，可选，对第dim个自变量求导，默认为0
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    
    derivative operator for multivariate function( same derivation to each component)
    Multivariate vector functions can be decomposed into several scalar functions
    Unable to recognize the two-dimensional list as an independent variable, loop is required
    
    Parameters
    ----------
    dim: int, callable, derivation of the dim independent variable, default=0
    acc: float, callable, accuracy of derivation, default=0.05
    '''
    def __init__(self, dim=0, acc=0.05):
        self.dim = dim
        self.acc = acc
    

    def fit(self, f):
        '''
        参数
        ----
        f:函数
        函数的自变量x是列表

        返回值：一个导函数
        
        
        Parameter
        ---------
        f: function
        argument x of function should be a list
        
        return: a derivative function
        '''
        def obj(x):
            x = np.array(x, dtype=float)
            x[self.dim] += self.acc
            func1 = f(x)
            if type(func1).__name__ == 'tuple' or 'list':
                func1 = np.array(func1)
            x[self.dim] -= 2 * self.acc
            func2 = f(x)
            if type(func2).__name__ == 'tuple' or 'list':
                func2 = np.array(func2)
            de = (func1 - func2) / (self.acc + self.acc)
            return de
        return obj



class Del():
    '''
    ▽算子
    被作用函数的自变量x是列表
    
    参数
    ----
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    
    nabla operator ▽
    the argument x of the affected function should be a list
    
    Parameters
    ----------
    acc: float, callable, accuracy of derivation, default=0.05
    '''
    def __init__(self, acc=0.05):
        self.acc = acc
    

    def grad(self, f, dim=None):
        '''
        求标量函数梯度：▽f(r)
        
        参数
        ----
        f：函数，要求函数f返回一个数值
        dim：整型，求导的维度，默认全部求导
        
        返回值：一个梯度函数
        
        
        gradient of scalar function: ▽f(r)
        
        Parameters
        ----------
        f: function, the function should return a number
        dim: int, dimensions for derivation, default: all
        
        return: a gradient function
        '''
        def obj(x):
            x = np.array(x, dtype=float)
            if not dim:
                dim = len(x)
            result = []
            for i in range(dim):
                x[i] += self.acc
                func1 = f(x)
                x[i] -= 2 * self.acc
                func2 = f(x)
                x[i] += self.acc
                de = (func1 - func2) / (self.acc + self.acc)
                result.append(de)
            result = np.array(result)
            return result
        return obj
    

    def dot(self, f):
        '''
        ▽点乘矢量函数：▽·f(r)
        
        参数
        ----
        f：函数，要求函数f返回一个列表

        返回值：一个新函数
        
        
        dot product between ▽ and vector function: ▽·f(r)
        
        Parameter
        ---------
        f: function, the function should return a list
        
        return: a new function
        '''
        def obj(x):
            x = np.array(x, dtype=float)
            result = []
            for i in range(len(x)):
                x[i] += self.acc
                func1 = f(x)[i]
                x[i] -= 2*self.acc
                func2 = f(x)[i]
                x[i] += self.acc
                de = (func1 - func2) / (self.acc + self.acc)
                result.append(de)
            result = np.array(result)
            return result
        return obj
    

    def __diff2forc(self, a, b, f, x):
        '''
        一个中介函数，方便后文程序使用
        '''
        x[b] += self.acc
        func1 = f(x)[a]
        x[b] -= 2*self.acc
        func2 = f(x)[a]
        x[b] += self.acc
        de = (func1 - func2) / (self.acc + self.acc)
        return de
    
    
    def cross(self, f):
        '''
        ▽叉乘矢量函数：▽×f(r)
        
        参数
        ----
        f：函数，要求函数是三维矢量函数

        返回值：一个新函数
        
        
        ▽ cross vector function: ▽×f(r)
        
        Parameter
        ---------
        f: function, the function should be a three-dimension vector function
        
        return: a new function
        '''
        def obj(x):
            x = np.array(x, dtype=float)
            result = np.array([
                Del.__diff2forc(self, 2, 1, f, x) -\
                Del.__diff2forc(self, 1, 2, f, x),
                Del.__diff2forc(self, 0, 2, f, x) -\
                Del.__diff2forc(self, 2, 0, f, x),
                Del.__diff2forc(self, 1, 0, f, x) -\
                Del.__diff2forc(self, 0, 1, f, x)])
            return result
        return obj



class Laplace():
    '''
    △算子 = ▽**2
    被作用函数的自变量x是列表
    
    参数
    ----
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    
    Laplace operator: △ = ▽**2
    the argument x of the affected function should be a list
    
    Parameter
    ---------
    acc: float, callable, accuracy of derivation, default=0.05
    '''
    def __init__(self, acc=0.05):
        self.acc = acc
    

    def fit(self, f):
        '''
        参数
        ----
        f：函数

        返回值：一个新函数
        
        
        Parameter
        ---------
        f: function
        
        return: a new function
        '''
        def obj(x):
            x = np.array(x, dtype=float)
            result = 0
            for i in range(len(x)):
                func1 = 2 * f(x)
                x[i] += 2 * self.acc
                func2 = f(x)
                x[i] -= 4 * self.acc
                func3 = f(x)
                x[i] += 2 * self.acc
                de = (func2 + func3 - 2 * func1) / 4 / self.acc**2
                result += de
            return result
        return obj



class H():
    '''
    哈密顿算符：H = - hr**2 / 2m * ▽**2 + U
    
    参数
    ----
    m：数，例子质量
    U：数或函数，势能
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    
    Hamilton: H = - hr**2 / 2m * ▽**2 + U
    
    Parameters
    ----------
    m: number, the mass of the particle
    U: number or function, potential energy
    acc: float, callable, accuracy of derivation, default=0.05
    '''
    def __init__(self, m, U, acc=0.05):
        self.m = m
        self.U = U
        self.acc = acc
    
    
    def fit(self, f):
        '''
        参数
        ----
        f：函数
        
        
        Parameter
        ---------
        f: function
        '''
        def obj(x):
            x = np.array(x, dtype=float)
            result = 0
            for i in range(len(x)):
                func1 = 2 * f(x)
                x[i] += 2 * self.acc
                func2 = f(x)
                x[i] -= 4 * self.acc
                func3 = f(x)
                x[i] += 2 * self.acc
                de = (func2 + func3 - 2 * func1) / 4 / self.acc**2
                result += de
            result *= (C.h / (2 * np.pi))**2 / (2 * self.m)
            
            if type(self.U).__name__ == 'function':
                result += self.U(x) * f(x)
            else:
                result += self.U * f(x)
            return result