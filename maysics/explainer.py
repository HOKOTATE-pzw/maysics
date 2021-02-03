'''
本模块用于评估和选择模型

This module is ued for estimating and selecting models
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import comb
plt.rcParams['axes.unicode_minus'] =False


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


class SHAP_and_Shapley():
    def __transform(self, b_list):
        # 二进制列表转十进制数
        d_value = 0
        for i in range(len(b_list)):
            d_value += b_list[i] * 2**i
        return d_value
    
    
    def _pre_fit(self, data, replace):
        dim = data.shape[1]
        shap_and_shapley_values = []
        
        # 求各个特征组合的预测值
        prediction = []
        for i in range(2**dim):
            new_data = data.copy()
            loc = np.array([i >>d & 1 for d in range(dim)][::1])
            loc = np.where(loc==0)[0]
            if len(loc) != 0:
                new_data[:, loc] = replace
            prediction_part = self.predict(new_data)
            prediction.append(prediction_part)
        prediction = np.array(prediction)
        
        # 求每个特征的SHAP值
        for i in range(dim):
            loc = np.array([2**i >>d & 1 for d in range(dim)][::1])
            loc = np.where(loc==1)[0][0]
            shap_value = 0
            
            # 每一层差值的加权求和
            for j in range(2**(dim-1)):
                b_list = [j >>d & 1 for d in range(dim)][::1]
                b_list.insert(loc, 0)
                b_value = SHAP_and_Shapley.__transform(self, b_list)
                # 1 / ((sum(b_list) + 1) * comb(dim, (sum(b_list) + 1)))是权重
                shap_value += (prediction[b_value + 2**i] - prediction[b_value]) / \
                              ((sum(b_list) + 1) * comb(dim, (sum(b_list) + 1)))
            shap_and_shapley_values.append(shap_value)
        return np.array(shap_and_shapley_values)
    
    
    def __image_process(self, labels, index, top):
        if not index:
            index = np.arange(0, self.values.shape[0], dtype=np.int)
        new_values = self.values[index]
        
        # 从大到小排序
        sort_list = np.argsort(new_values)
        new_values = np.sort(new_values)
        
        if not labels:
            labels = sort_list
        else:
            labels = np.array(labels, dtype=np.str)
            labels = labels[sort_list]
        
        if not top:
            top = self.values.shape[0]
        new_values = new_values[-top:]
        labels = labels[-top:]
        
        width = new_values.shape[0] * 0.5
        fig = plt.figure(figsize=(8, width))
        ax = fig.add_subplot(1, 1, 1)
        color = []
        for i in range(new_values.shape[0]):
            if new_values[i] >= 0:
                color.append('#FFE100')
            else:
                color.append('#1E90FF')
        
        rows = ax.barh(range(new_values.shape[0]), new_values, color=color)
        for rect in rows:
            w = rect.get_width()
            if w >= 0:
                ha = 'left'
            else:
                ha = 'right'
            ax.text(w, rect.get_y()+rect.get_height()/2, w, ha=ha, va='center')
        
        ax.set_yticks(range(new_values.shape[0]))
        ax.set_yticklabels(labels)
        plt.xticks(())
    
    
    def show(self, labels=None, index=None, top=None):
        '''
        作图并显示
        
        参数
        ----
        labels：一维列表，可选，特征名称，默认为None
        index：一维列表，可选，特征索引，默认为None，表示全选
        top：整型，可选，表示显示值最高的前top个特征，默认为None，表示全选
        
        
        Display the image
        
        Parameters
        ----------
        labels: 1-D list, callable, names of features, default=None
        index: 1-D list, callable, index of features, default=None, which means select all
        top: int, callable, display "top" features with the highest values, default=None, which means select all
        '''
        SHAP_and_Shapley.__image_process(self, labels, index, top)
        plt.show()
    
    
    def savefig(self, filename, labels=None, index=None, top=None):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串类型，文件名
        labels：一维列表，可选，特征名称，默认为None
        index：一维列表，可选，特征索引，默认为None，表示全选
        top：整型，可选，表示显示值最高的前top个特征，默认为None，表示全选
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        labels: 1-D list, callable, names of features, default=None
        index: 1-D list, callable, index of features, default=None, which means select all
        top: int, callable, display "top" features with the highest values, default=None, which means select all
        '''
        SHAP_and_Shapley.__image_process(self, labels, index, top)
        plt.savefig(filename)


class SHAP(SHAP_and_Shapley):
    def __init__(self, predict):
        '''
        计算局部点在模型中的SHAP值
        
        参数
        ----
        predict：函数类型，模型的预测函数
        
        属性
        ----
        values：一维ndarray，每个特征的SHAP值
        
        
        Calculate the SHAP of local points in the model
        
        Parameter
        ---------
        predict: function, the predict function of the model
        
        Attribute
        ---------
        values: 1-D ndarray, SHAP of each feature
        '''
        self.predict = predict
    
    
    def fit(self, data, replace=0):
        '''
        计算SHAP值
        
        参数
        ----
        data：一维数组，局部点
        replace：数或函数类型，可选，特征的替换值，函数须以np.array(data)为输入，默认为0
        
        
        Calculate the SHAP values
        
        Parameters
        ----------
        data: 1-D array, local point
        replace: num or function, callable, replacement values of features, for function, np.array(data) is the input, default=0
        '''
        data = np.array(data)
        # 确定替换值
        if type(replace).__name__ == 'function':
            replace = replace(data)
        data = np.array([data])
        
        self.values = SHAP_and_Shapley._pre_fit(self, data, replace).reshape(-1)


class Shapley(SHAP_and_Shapley):
    def __init__(self, predict):
        '''
        计算样本集在模型中的Shapley值
        
        参数
        ----
        predict：函数类型，模型的预测函数
        
        属性
        ----
        values：一维ndarray，每个特征的Shapley值
        
        
        Calculate the Shapley values of data set in the model
        
        Parameter
        ---------
        predict: function, the predict function of the model
        
        Attribute
        ---------
        values: 1-D ndarray, Shapley value of each feature
        '''
        self.predict = predict
    
    
    def fit(self, data, replace=0):
        '''
        计算Shapley
        
        参数
        ----
        data：二维数组，数据集
        replace：数或函数类型，可选，特征的替换值，函数须以np.array(data)为输入，默认为0
        
        
        Calculate the Shapley
        
        Parameters
        ----------
        data: 2-D array, data set
        replace: num or function, callable, replacement values of features, for function, np.array(data) is the input, default=0
        '''
        data = np.array(data)
        # 确定替换值
        if type(replace).__name__ == 'function':
            replace = replace(data)
        
        self.values = SHAP_and_Shapley._pre_fit(self, data, replace).T
        self.values = self.values.mean(axis=0)