'''
本模块储存着常用算符和积分方法
调用模块中的算符时应注意：
1、算符作用的多元函数的自变量需为列表(list、tuple、ndarray均可，下同)
2、算符作用的标量函数的输出值需为单个值
3、算符作用的矢量函数的输出值需为列表

This module stores the common operators and integration method
When calling operators in the module, pay attention to:
1. The argument of the multivariate function acted by the operators should be list (list, tuple, ndarray, the same below)
2. The output value of scalar function acted by the operators should be a single value
3. The output value of vector function acted by the operators should be list
'''

import numpy as np
from scipy import constants as C
from maysics.utils import grid_net
from matplotlib import pyplot as plt


def lim(func, x0, acc=0.01, method='both'):
    '''
    求极限
    lim x→x0+ f(x) ≈ f(x+dx)
    lim x→x0- f(x) ≈ f(x-dx)
    lim x→x0 ≈ (f(x+dx) + f(x-dx)) / 2
    
    参数
    ----
    func：函数类型
    x0：取极限的点
    acc：浮点数类型，可选，极限精度，即dx，默认为0.01
    method：字符串类型，求极限方法，可选'both'、'rigth'或'+'(右极限)、'left'或'-'(左极限)
    
    
    Calculate the limit values
    lim x→x0+ f(x) ≈ f(x+dx)
    lim x→x0- f(x) ≈ f(x-dx)
    lim x→x0 ≈ (f(x+dx) + f(x-dx)) / 2
    
    Parameters
    ----------
    func: function
    x0: point of limit
    acc: float, callable, the accuracy of calculation, equals to dx, default=0.01
    method: str, the method of calculation, 'both', 'right' or '+'(right limit), 'left' or '-'(left limit) are optional
    '''
    if method == 'both':
        func1 = func(x0 + acc)
        func2 = func(x0 - acc)
        return 0.5 * (func1 + func2)
    
    elif method == '+' or method == 'right':
        return func(x0 + acc)
    
    elif method == '-' or method == 'left':
        return func(x0 - acc)
    
    else:
        raise Exception("Parameter 'method' must be one of 'both', 'right', '+', 'left', '-'.")



def dif(f, dim=0, acc=0.05):
    '''
    对多元函数的求导算子，对矢量函数则相当于对每个分量函数都进行了相同的求导
    多元矢量函数可以分解为多个标量函数
    无法将二维列表识别成一行作为一个自变量，需要循环
    
    参数
    ----
    f:函数类型, 函数的自变量x是列表
    dim：整型，可选，对第dim个自变量求导，默认为0
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    返回
    ----
    导函数
    
    
    derivative operator for multivariate function( same derivation to each component)
    Multivariate vector functions can be decomposed into several scalar functions
    Unable to recognize the two-dimensional list as an independent variable, loop is required
    
    Parameters
    ----------
    f: function, argument x of function should be a list
    dim: int, callable, derivation of the dim independent variable, default=0
    acc: float, callable, accuracy of derivation, default=0.05
    
    Return
    ------
    derivative function
    '''
    def obj(x):
        x = np.array(x, dtype=float)
        x[dim] += acc
        func1 = f(x)
        if type(func1).__name__ == 'tuple' or 'list':
            func1 = np.array(func1)
        x[dim] -= 2 * acc
        func2 = f(x)
        if type(func2).__name__ == 'tuple' or 'list':
            func2 = np.array(func2)
        de = (func1 - func2) / (acc + acc)
        return de
    return obj



def ha(f, m, U, acc=0.05):
    '''
    哈密顿算符：ha = - hr**2 / 2m * ▽**2 + U
    
    参数
    ----
    f：函数
    m：数，例子质量
    U：数或函数，势能
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    返回
    ----
    函数
    
    
    Hamilton: ha = - hr**2 / 2m * ▽**2 + U
    
    Parameters
    ----------
    m: number, the mass of the particle
    U: number or function, potential energy
    acc: float, callable, accuracy of derivation, default=0.05
    f: function
    
    Return
    ------
    function
    '''
    def obj(x):
        x = np.array(x, dtype=float)
        result = 0
        for i in range(len(x)):
            func1 = 2 * f(x)
            x[i] += 2 * acc
            func2 = f(x)
            x[i] -= 4 * acc
            func3 = f(x)
            x[i] += 2 * acc
            de = (func2 + func3 - 2 * func1) / 4 / acc**2
            result += de
        result *= (C.h / (2 * np.pi))**2 / (2 * m)
        
        if type(U).__name__ == 'function':
            result += U(x) * f(x)
        else:
            result += U * f(x)
        return result
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
        
        返回
        ----
        梯度函数▽f
        
        
        gradient of scalar function: ▽f(r)
        
        Parameters
        ----------
        f: function, the function should return a number
        dim: int, dimensions for derivation, default: all
        
        Return
        ------
        gradient function ▽f
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

        返回
        ----
        函数▽·f
        
        
        dot product between ▽ and vector function: ▽·f(r)
        
        Parameter
        ---------
        f: function, the function should return a list
        
        Return
        ------
        function ▽·f
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

        返回
        ----
        函数 ▽×f
        
        
        ▽ cross vector function: ▽×f(r)
        
        Parameter
        ---------
        f: function, the function should be a three-dimension vector function
        
        Return
        ------
        function ▽×f
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


    def laplace(self, f):
        '''
        △算子 = ▽**2
        被作用函数的自变量x是列表
        
        参数
        ----
        f：函数
        
        返回
        ----
        函数△f
        
        
        Laplace operator: △ = ▽**2
        the argument x of the affected function should be a list
        
        Parameter
        ---------
        f: function
        
        Return
        ------
        function △f
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



class Inte():
    def __init__(self, method='rect', dim=1):
        '''
        定积分
        
        矩形法rect：
        S ≈ ∑ f(xi)dv
        dv = ∏ dx ≈ ∏ Δx = ∏ acc
        
        蒙特卡洛法mc：
        在区域V = area * height中产生loop个随机点
        计算符合条件的随机点个数effect_points
        S ≈ V * effect_points / loop
        
        参数
        ----
        method：字符串类型，可选'rect'、'mc'，默认为'rect'
        dim：整型，1表示被积函数的输入为1维数组，适用于普通自定义函数，2表示被积函数的输入为2维数组，适用于机器学习模型，默认为1
        
        属性
        ----
        history：数组类型，储存每一次的积分结果
        
        
        Definite integral
        
        ractangle method ('rect')：
        S ≈ ∑ f(xi)dv
        dv = ∏ dx ≈ ∏ Δx = ∏ acc
        
        MonteCarlo method ('mc')：
        generate 'loop' random points in area 'V = area * height'
        calculate the number of qualified points 'effect_points'
        S ≈ V * effect_points / loop
        
        Parameters
        ----------
        method: str, callable, 'rect' and 'mc' are optional, default='rect'
        dim: int, 1 means the input of integrand is 1-D list, like normal functions, 2 means the input of integrand is 2-D list, like machine learning models, default=1
        
        Attribute
        ---------
        history: list, stores every integral result
        '''
        self.method = method
        self.__dim = dim
        self.history = []
        
        if dim > 2:
            raise Exception("Parameter 'dim' must be one of 1 and 2.")
        
        if method != 'rect' and method != 'mc':
            raise Exception("Parameter 'method' must be one of 'rect' and 'mc'.")
    
    
    def __mc_fit(self, func, area, condition, param, loop, height, random_state):
        '''
        蒙特卡洛法
        
        
        MonteCarlo method
        '''
        np.random.seed(random_state)
        area = np.array(area)
        area_length = area[:, 1] - area[:, 0]
        V = area_length.prod() * height
        
        num = len(area_length)
        area_points = np.random.rand(loop, num) * area_length + area[:, 0]
        func_points = np.random.rand(loop) * height
        
        effect_points = 0
        if self.__dim == 1:
            if not condition:
                for i in range(loop):
                    if func_points[i] <= func(area_points[i]):
                        effect_points += 1
            
            else:
                for i in range(loop):
                    if func_points[i] <= func(area_points[i]) and condition(area_points[i], **param):
                        effect_points += 1
        
        elif self.__dim == 2:
            func_points_2 = func(area_points)
            if not condition:
                for i in range(loop):
                    if func_points[i] <= func_points_2[i]:
                        effect_points += 1
            
            else:
                for i in range(loop):
                    if func_points[i] <= func_points_2[i] and condition(area_points[i], **param):
                        effect_points += 1
        
        effect_points_rate = effect_points / loop
        
        return V * effect_points_rate
    
    
    def __rect_fit(self, func, area, condition, param, acc):
        '''
        矩形法
        
        
        Rectangle method
        '''
        acc = np.array(acc)
        dv = acc.prod()
        
        points_net = []
        for i in range(len(area)):
            points_net.append(np.arange(area[i][0], area[i][1], acc[i]))
        points_net = grid_net(*points_net)
        
        if self.__dim == 1:
            func_points=[]
            if not condition:
                for i in points_net:
                    func_points.append(func(i))
            
            else:
                for i in points_net:
                    if condition(i, **param):
                        func_points.append(func(i))
            
            func_points = np.array(func_points) * dv
        
        elif self.__dim == 2:
            points_net = points_net.tolist()
            if condition:
                for i in points_net[:]:
                    if not condition(i, **param):
                        points_net.remove(i)
            points_net = np.array(points_net)
            func_points = func(points_net) * dv
        
        return func_points.sum()
    
    
    def fit(self, func, area, condition=None, param={}, acc=0.1, loop=10000, height=1, dim=1, random_state=None):
        '''
        积分
        
        参数
        ----
        func：函数类型，被积函数
        area：二维列表，积分区域，由自变量上下限列表组成
            如：积分区域为[a, b]时，area=[[a, b]]
                积分区域为二维区域x1∈[a1, b1]，x2属于[a2, b2]时，area=[[a1, b1], [a2, b2]]
        condition：函数类型，可选，条件函数，符合条件的输出Ture，否则输出False，条件函数的第一个参数的输入须为1维数组
        param：字典类型，可选，当条件函数有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        acc：浮点数或列表类型，可选，积分精度，只有method='rect'时才有效，默认为0.1
        loop：整型，可选，产生的随机数数目，只有method='mc'时才有效，默认为10000
        height：浮点数类型，可选，高度，只有method='mc'时才有效，默认为1
        random_state：整型，随机种子，只有method='mc'时才有效
        
        
        Intergration
        
        Parameters
        ----------
        func: function, integrand
        area: 2-D list, integral region, composed of a list of upper and lower limits of independent variables
            e.g. when the integral region is [a, b], area = [[a, b]]
                 when the 2-D integral region is x1∈[a1, b1] and x2∈[a2, b2], area=[[a1, b1], [a2, b2]]
        condition: function, callable, condition function with the input of the first parameter as 1-D list, if input if qualified, ouput True, otherwise output False
        param: dict, callable, when condtition function has other non-default parameters, 'param' needs to be input a dictionary with parm_name as key and param_value as value, an empty dict to default
        acc: float or list, callable, integration accuracy, it's effective only when method='rect', default=0.1
        loop: int, callable, the number of generated random numbers, it's effective only when method='mc', default=10000
        height: float, callable, height, it's effective only when method='mc', default=1
        random_state: int, random seed, it's effective only when method='mc'
        '''
        if self.method == 'rect':
            if type(acc).__name__ == 'float' or type(acc).__name__ == 'int':
                acc = np.ones(len(area)) * acc
            self.history.append(Inte.__rect_fit(self, func=func, area=area, condition=condition, param=param, acc=acc))
        
        elif self.method == 'mc':
            self.history.append(Inte.__mc_fit(self, func=func, area=area, condition=condition, param=param,
                                              loop=loop, height=height, random_state=random_state))
    
    
    def clr(self):
        '''
        清空history
        
        
        Clear the history
        '''
        self.history = []
    
    
    def show(self, x=None, scatter=False):
        '''
        作积分值图并显示
        
        参数
        ----
        x：列表类型，可选，自变量
        scatter：布尔类型，可选，True代表绘制散点图，False代表绘制折线图，默认为False
        
        
        Display the integral value image
        
        Parameters
        ----------
        x: list, callable, independent variable
        scatter: bool, callable, 'True' means drawing the scatter image, 'False' means drawing the line image, default=False
        '''
        if not x:
            x = np.arange(len(self.history))
        
        if not scatter:
            plt.plot(x, self.history)
        else:
            plt.scatter(x, self.history)
        
        plt.show()
    
    
    def savefig(self, filename, x=None, scatter=False):
        '''
        作积分值图并储存
        
        参数
        ----
        filename：字符串类型，文件名
        x：列表类型，可选，自变量
        scatter：布尔类型，可选，True代表绘制散点图，False代表绘制折线图，默认为False
        
        
        Save the integral value image
        
        Parameters
        ----------
        filename: str, file name
        x: list, callable, independent variable
        scatter: bool, callable, 'True' means drawing the scatter image, 'False' means drawing the line image, default=False
        '''
        if not x:
            x=np.arange(len(self.history))
        
        if not scatter:
            plt.plot(x, self.history)
        else:
            plt.scatter(x, self.history)
        
        plt.savefig(filename)