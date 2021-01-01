'''
本模块用于评估和选择模型

This module is ued for estimating and selecting models
'''
import numpy as np
from matplotlib import pyplot as plt


class Error():
    '''
    误差分析
    
    参数
    ----
    func：函数类型，模型的预测函数
    
    属性
    ----
    abs_error：1-D ndarray数组，绝对误差列表
    rel_error：1-D ndarray数组，相对误差列表
    abs_sort：绝对误差从大到小的排序
    rel_sort：相对误差从大到小的排序
    mae：平均绝对误差
    mape：平均绝对百分比误差
    mse：平均平方误差
    rmse：均方根误差
    sae：绝对误差和
    sse：残差平方和
    
    
    Error Analysis
    
    Parameter
    ---------
    func: function, predicting function of models
    
    Atrributes
    ----------
    abs_error: 1-D ndarray, absolute error list
    rel_error: 1-D ndarray, relative error list
    abs_sort: list of absolute values of errors sorted from large to small
    rel_sort: list of relative values of errors sorted from large to small
    mae: mean absolute error
    mape: mean absolute percentage error
    mse: mean squared error
    rmse: root mean square error
    sae: sum of absolute error
    sse: sum of squared error
    '''
    def __init__(self, func):
        self.__func = func
    
    
    def fit(self, data, target):
        data = np.array(data)
        target = np.array(target)
        self.__num_data = len(target)
        
        predict_target = self.__func(data)
        self.abs_error = abs(target - predict_target)
        self.rel_error = abs((target - predict_target) / target)
    
    @property
    def abs_sort(self):
        error_index = np.arange(self.__num_data)
        return sorted(list(zip(self.abs_error, error_index)), reverse=True)
    
    @property
    def rel_sort(self):
        error_index = np.arange(self.__num_data)
        return sorted(list(zip(self.rel_error, error_index)), reverse=True)
    
    @property
    def sse(self):
        return sum(self.abs_error**2)
    
    @property
    def sae(self):
        return sum(self.abs_error)
    
    @property
    def mse(self):
        return sum(self.abs_error**2) / self.__num_data
    
    @property
    def mae(self):
        return sum(self.abs_error) / self.__num_data
    
    @property
    def rmse(self):
        return (sum(self.abs_error**2) / self.__num_data)**0.5
    
    @property
    def mape(self):
        return sum(self.rel_error) / self.__num_data


class Sense():
    '''
    灵敏度分析
    r = (x0, x1, x2, ..., xn)
    y = f(r)
    第i个特征在r=r0时的灵敏度：
    s(xi, r0)= (dy/dxi) * (xi/y)   (r=r0)
    
    参数
    ----
    func：函数类型，模型的预测函数，若函数需要输入列表，则列表须为ndarray
    acc：浮点数类型，可选，求导的精度，默认为0.1
    
    属性
    ----
    s_mat：由特征的灵敏度值组成的矩阵
    prediction：预测值列表
    
    
    Sensitivity Analysis
    r = (x0, x1, x2, ..., xn)
    y = f(r)
    the sensitivity of the ith feature at r=r0:
    s(xi, r0)= (dy/dxi) * (xi/y)   (r=r0)
    
    Parameters
    ----------
    func: function, predicting function of models, if the function requires a list as input, the list must be ndarray
    acc: float, callable, accuracy of derivation, default=0.1
    
    Attributes
    ----------
    s_mat: matrix combined of sensitivities of features
    prediction: list of predicted values
    '''
    def __init__(self, func, acc=0.1):
        self.__func = func
        self.acc = acc
        self.s_mat = []
        self.prediction = []
    
    
    def fit(self, x0):
        '''
        参数
        ----
        x0：数，1-D或2-D ndarray，特征的初始值，不支持批量输入
        
        
        Parameter
        ---------
        x0: num, 1-D or 2-D ndarray, initial values of features, batch input is not supported
        '''
        x0 = np.array(x0, dtype=np.float)
        acc2 = 0.5 * self.acc
        func0 = self.__func(x0)
        self.prediction.append(func0)
        s_list = []
        
        if len(x0.shape) == 0:
            x0 += acc2
            func1 = self.__func(x0)
            x0 -= self.acc
            func2 = self.__func(x0)
            s_list = (func1 - func2) / (self.acc * func0) * x0
        
        elif len(x0.shape) == 1:
            for i in range(x0.shape[0]):
                x0[i] += acc2
                func1 = self.__func(x0)
                x0[i] -= self.acc
                func2 = self.__func(x0)
                de = (func1 - func2) / (self.acc * func0) * x0[i]
                x0[i] += acc2
                s_list.append(de)
        
        elif len(x0.shape) == 2:
            for i in range(x0.shape[1]):
                x0[0][i] += acc2
                func1 = self.__func(x0)
                x0[0][i] -= self.acc
                func2 = self.__func(x0)
                de = (np.array([func1]).T - np.array([func2]).T) / (self.acc * np.array(func0).T) * x0[0][i]
                x0[0][i] += acc2
                s_list.append(de)
            
        self.s_mat.append(s_list)
    
    
    def clr(self):
        '''
        清空s_mat和prediction
        
        
        Clear the s_mat and the prediction
        '''
        self.s_mat = []
        self.prediction = []