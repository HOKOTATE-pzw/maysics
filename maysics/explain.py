'''
本模块用于评估和选择模型

This module is ued for estimating and selecting models
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import comb
plt.rcParams['axes.unicode_minus'] =False


def _image_detail(data, labels, index, top, estimate, n1, n2):
    if not index:
        index = np.arange(0, data.shape[0], dtype=int)
    new_data = data[index]
    
    # 从大到小排序
    sort_list = np.argsort(new_data)
    new_data = np.sort(new_data)
    
    if not labels:
        labels = sort_list.astype(str)
    else:
        labels = np.array(labels, dtype=str)
        labels = labels[sort_list]
    
    if not top:
        top = data.shape[0]
    new_data = new_data[-top:]
    labels = labels[-top:]
    
    width = new_data.shape[0] * 0.5 * n1
    fig = plt.figure(figsize=(8, width))
    ax = fig.add_subplot(n1, 1, n2)
    
    color = []
    for i in range(new_data.shape[0]):
        if new_data[i] >= 0:
            color.append('#FFE100')
        else:
            color.append('#1E90FF')
    
    rows = ax.barh(range(new_data.shape[0]), new_data, color=color)
    for rect in rows:
        w = rect.get_width()
        if w >= 0:
            ha = 'left'
        else:
            ha = 'right'
        if not estimate:
            new_w = w
        else:
            new_w = np.around(w, estimate)
        ax.text(w, rect.get_y()+rect.get_height()/2, new_w, ha=ha, va='center')
    
    ax.set_yticks(range(new_data.shape[0]))
    ax.set_yticklabels(labels)
    plt.xticks(())


def _image_process(data, labels, index, top, estimate):
    if len(data.shape) == 1:
        _image_detail(data, labels, index, top, estimate, 1, 1)
    elif len(data.shape) == 2:
        num = data.shape[0]
        for j in range(num):
            _image_detail(data[j], labels, index, top, estimate, num, j+1)


def abs_error(func, data, target, param={}):
    '''
    绝对误差列表
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数组，绝对误差列表
    
    
    Absolute Error List
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    array, absolute error list
    '''
    data = np.array(data)
    target = np.array(target)
    predict_target = func(data, **param)
    return abs(target - predict_target)


def rel_error(func, data, target, param={}):
    '''
    相对误差列表
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数组，相对误差列表
    
    
    Relative Error List
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    array, relative error list
    '''
    data = np.array(data)
    target = np.array(target)
    predict_target = func(data, **param)
    return abs((target - predict_target) / target)


def abs_sort(func, data, target, param={}):
    '''
    绝对误差从大到小的排序
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    列表，绝对误差从大到小的排序
    
    
    List of Absolute Values of Errors Sorted from Large to Small
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    list, list of absolute values of errors sorted from large to small
    '''
    error_index = np.arange(len(target))
    return sorted(list(zip(abs_error(func, data, target, param=param), error_index)), reverse=True)


def rel_sort(func, data, target, param={}):
    '''
    相对误差从大到小的排序
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    列表，绝对误差从大到小的排序
    
    
    List of Relative Values of Errors Sorted from Large to Small
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    list, list of relative values of errors sorted from large to small
    '''
    error_index = np.arange(len(target))
    return sorted(list(zip(rel_error(func, data, target, param=param), error_index)), reverse=True)


def sse(func, data, target, param={}):
    '''
    残差平方和
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数类型，残差平方和
    
    
    Sum of Squared Error
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    num, sum of squared error
    '''
    return sum(abs_error(func, data, target, param=param)**2)


def sae(func, data, target, param={}):
    '''
    绝对误差和
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数类型，绝对误差和
    
    
    Sum of Absolute Error
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    num, sum of absolute error
    '''
    return sum(abs_error(func, data, target, param=param))


def mse(func, data, target, param={}):
    '''
    平均平方误差
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数类型，平均平方误差
    
    
    Mean Squared Error
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    num, mean squared error
    '''
    data = np.array(data)
    target = np.array(target)
    predict_target = func(data, *param)
    return sum(abs(target - predict_target)**2) / len(target)


def mae(func, data, target, param={}):
    '''
    平均绝对误差
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数类型，平均绝对误差
    
    
    Mean Absolute Error
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    num, mean absolute error
    '''
    return sum(abs_error(func, data, target, param=param)) / len(target)


def rmse(func, data, target, param={}):
    '''
    均方根误差
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数类型，均方根误差
    
    
    Root Mean Square Error
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    num, root mean square error
    '''
    return (sum(abs_error(func, data, target, param=param)**2) / len(target))**0.5


def mape(func, data, target, param={}):
    '''
    平均绝对百分比误差
    
    参数
    ----
    func：函数类型，模型的预测函数
    data：ndarray，数据集的自变量
    target：一维ndarray，数据集的因变量
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    数类型，平均绝对百分比误差
    
    
    Mean Absolute Percentage Error
    
    Parameters
    ----------
    func: function, predicting function of models
    data: ndarray, independent variable of data set
    target: ndarray, dependent variable of data set
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    num, mean absolute percentage error
    '''
    return sum(rel_error(func, data, target, param=param)) / len(target)


def sense(func, x0, acc=0.1, param={}):
    '''
    灵敏度分析
    r = (x0, x1, x2, ..., xn)
    y = f(r)
    第i个特征在r=r0时的灵敏度：
    s(xi, r0)= (dy/dxi) * (xi/y)   (r=r0)
    
    参数
    ----
    func：函数类型，模型的预测函数，若函数需要输入数组，则数组须为ndarray
    x0：数，一维或二维ndarray，与func的输入格式相同，特征的初始值
    acc：浮点数类型，可选，求导的精度，默认为0.1
    param：字典类型，可选，当func有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    各个维度的灵敏度，格式与func的输出格式相同
    
    
    Sensitivity Analysis
    r = (x0, x1, x2, ..., xn)
    y = f(r)
    the sensitivity of the ith feature at r=r0:
    s(xi, r0)= (dy/dxi) * (xi/y)   (r=r0)
    
    Parameters
    ----------
    func: function, predicting function of models, if the function requires a list as input, the list must be ndarray
    x0: num, 1D or 2D ndarray, the format is the same as the input of func, initial values of features
    acc: float, callable, accuracy of derivation, default=0.1
    param: dict, callable, When func has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    sensitivity of each dimension, whose format is the same as the output of func
    '''
    x0 = np.array(x0, dtype=float)
    acc2 = 0.5 * acc
    func0 = func(x0, **param)
    s_list = []
    
    if len(x0.shape) == 0:
        x0 += acc2
        func1 = func(x0, **param)
        x0 -= acc
        func2 = func(x0, **param)
        return (func1 - func2) / (acc * func0) * x0
    
    elif len(x0.shape) == 1:
        for i in range(x0.shape[0]):
            x0[i] += acc2
            func1 = func(x0, **param)
            x0[i] -= acc
            func2 = func(x0, **param)
            de = (func1 - func2) / (acc * func0) * x0[i]
            x0[i] += acc2
            s_list.append(de)
        s_list = np.array(s_list)
    
    elif len(x0.shape) == 2:
        for i in range(x0.shape[1]):
            x0[:, i] += acc2
            func1 = func(x0, **param)
            x0[:, i] -= acc
            func2 = func(x0, **param)
            de = (func1 - func2) / (acc * func0) * x0[:, i]
            x0[:, i] += acc2
            s_list.append(de)
        s_list = np.array(s_list).T
    return s_list


class SHAP_and_Shapley():
    def __transform(self, b_list):
        # 二进制列表转十进制数
        d_value = 0
        for i in range(len(b_list)):
            d_value += b_list[i] * 2**i
        return d_value
    
    
    def _pre_fit(self, data, replace, param):
        dim = data.shape[1]
        shap_and_shapley_values = []
        
        # 求各个特征组合的预测值
        prediction = []
        for i in range(2**dim):
            new_data = data.copy()
            loc = np.array([i >>d & 1 for d in range(dim)])
            loc = np.where(loc==0)[0]
            if len(loc) != 0:
                new_data[:, loc] = replace
            prediction_part = self.predict(new_data, **param)
            prediction.append(prediction_part)
        prediction = np.array(prediction)
        
        # 求每个特征的SHAP值
        for i in range(dim):
            loc = np.array([2**i >>d & 1 for d in range(dim)])
            loc = np.where(loc==1)[0][0]
            shap_value = 0
            
            # 每一层差值的加权求和
            for j in range(2**(dim-1)):
                b_list = [j >>d & 1 for d in range(dim)]
                b_list.insert(loc, 0)
                b_value = self.__transform(b_list)
                # 1 / ((sum(b_list) + 1) * comb(dim, (sum(b_list) + 1)))是权重
                shap_value += (prediction[b_value + 2**i] - prediction[b_value]) / \
                              ((sum(b_list) + 1) * comb(dim, (sum(b_list) + 1)))
            shap_and_shapley_values.append(shap_value)
        return np.array(shap_and_shapley_values)
    
    
    def show(self, labels=None, index=None, top=None, estimate=None):
        '''
        作图并显示
        
        参数
        ----
        labels：一维列表，可选，特征名称，默认为None
        index：一维列表，可选，特征索引，默认为None，表示全选
        top：整型，可选，表示显示值最高的前top个特征，默认为None，表示全选
        estimate：整型，可选，表示图示示数保留的小数位数，默认为None
        
        
        Display the image
        
        Parameters
        ----------
        labels: 1D list, callable, names of features, default=None
        index: 1D list, callable, index of features, default=None, which means select all
        top: int, callable, display "top" features with the highest values, default=None, which means select all
        estimate: int, callable, indicating the number of decimal places reserved for the graphic display, default=None
        '''
        _image_process(self.values, labels, index, top, estimate)
        plt.show()
    
    
    def savefig(self, filename, labels=None, index=None, top=None, estimate=None):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串类型，文件名
        labels：一维列表，可选，特征名称，默认为None
        index：一维列表，可选，特征索引，默认为None，表示全选
        top：整型，可选，表示显示值最高的前top个特征，默认为None，表示全选
        estimate：整型，可选，表示图示示数保留的小数位数，默认为None
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        labels: 1D list, callable, names of features, default=None
        index: 1D list, callable, index of features, default=None, which means select all
        top: int, callable, display "top" features with the highest values, default=None, which means select all
        estimate: int, callable, indicating the number of decimal places reserved for the graphic display, default=None
        '''
        _image_process(self.values, labels, index, top, estimate)
        plt.savefig(filename)


class SHAP(SHAP_and_Shapley):
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
    values: 1D ndarray, SHAP of each feature
    '''
    def __init__(self, predict):
        self.predict = predict
    
    
    def fit(self, data, replace=0, param={}):
        '''
        计算SHAP值
        
        参数
        ----
        data：一维数组，局部点
        replace：数或函数类型，可选，特征的替换值，函数须以np.array(data)为输入，默认为0
        param：字典类型，可选，当predict有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        
        
        Calculate the SHAP values
        
        Parameters
        ----------
        data: 1D array, local point
        replace: num or function, callable, replacement values of features, for function, np.array(data) is the input, default=0
        param: dict, callable, When predict has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
        '''
        data = np.array(data)
        # 确定替换值
        if type(replace).__name__ == 'function':
            replace = replace(data)
        data = np.array([data])
        
        self.values = SHAP_and_Shapley._pre_fit(self, data, replace, param).reshape(-1)


class Shapley(SHAP_and_Shapley):
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
    values: 1D ndarray, Shapley value of each feature
    '''
    def __init__(self, predict):
        self.predict = predict
    
    
    def fit(self, data, replace=0, param={}):
        '''
        计算Shapley
        
        参数
        ----
        data：二维数组，数据集
        replace：数或函数类型，可选，特征的替换值，函数须以np.array(data)为输入，默认为0
        param：字典类型，可选，当predict有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        
        
        Calculate the Shapley
        
        Parameters
        ----------
        data: 2D array, data set
        replace: num or function, callable, replacement values of features, for function, np.array(data) is the input, default=0
        param: dict, callable, When predict has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
        '''
        data = np.array(data)
        # 确定替换值
        if type(replace).__name__ == 'function':
            replace = replace(data)
        
        self.values = SHAP_and_Shapley._pre_fit(self, data, replace, param).T
        self.values = self.values.mean(axis=0)


class Lime():
    '''
    局部代理
    对局部点进行扰动并输入模型得到新的数据集
    以新数据集为基础，拟合出新的线性模型AX+b用于局部代替黑盒模型
    利用该线性模型对黑盒模型进行解释
    预测值 = Σweight + intercept
    
    参数
    ----
    predict_f：函数类型，原模型的预测函数，仅支持批量输入函数
    
    属性
    ----
    coef：ndarray，线性模型的系数
    intercept：数或一维ndarray，线性模型的截距
    weight：一维ndarray，各个特征对预测值的贡献
    
    
    Local Interpretable Model-agnostic Explanation
    perturb the local points and put them into the model to get a new data set
    fit a linear model Ax + b to replace the black box model base on the new data set
    explain the black box model with the linear model
    prediction = Σweight + intercept
    
    Parameter
    ---------
    predict_f: function, the predict function of the model, only batch input functions are supported
    
    Attributes
    ----------
    coef: ndarray, coeficcient of linear model
    intercept: num or 1D ndarray, intercept of linear model
    weight: 1D ndarray, contribution to the prediction of each feature
    '''
    def __init__(self, predict_f):
        self.predict_f = predict_f
    
    
    def fit(self, data, acc=0.1, num=100, param={}, random_state=None):
        '''
        进行Lime计算
        
        参数
        ----
        data：数组，局部点
        acc：浮点数类型，可选，邻域范围，默认为0.1
        num：整型，可选，在领域抽样点的数量，默认为100
        param：字典类型，可选，当predict_f有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        random_state：整型，可选，随机种子
        
        
        Calculate
        
        Parameters
        ----------
        data: ndarray, local point
        acc: float, callable, neighborhood range， default=0.1
        num: int, callable
        param: dict, callable, When predict_f has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
        random_state: int, callable, random seed
        '''
        data = np.array(data)
        np.random.seed(random_state)
        X_set = np.random.rand(num, *data.shape) * 2 * acc - acc
        X_set += data
        y_set = self.predict_f(X_set, **param)
        shape = X_set.shape
        X_set = X_set.reshape(shape[0], -1)
        X_set = np.hstack((X_set, np.ones((X_set.shape[0], 1))))
        A = np.linalg.lstsq(X_set, y_set, rcond=None)[0]
        
        self.intercept = A[-1]
        self.coef = A[:-1].T
        self.weight = self.coef * data
    
    
    def predict(self, data):
        '''
        利用近似的线性模型进行预测
        
        参数
        ----
        data：小批量输入数组，需要预测的数据
        
        
        Use approximate linear model to predict
        
        Parameter
        ---------
        data: barch input ndarray, input data
        '''
        if len(self.intercept.shape) == 0:
            result = data * self.coef
            retult = result.reshape(result.shape[0], -1)
            result = result.sum(axis=1) + self.intercept
            return result
        
        else:
            all_result = []
            for i in range(self.coef.shape[0]):
                result = data * self.coef[i]
                result = result.reshape(result.shape[0], -1)
                result = result.sum(axis=1) + self.intercept[i]
                all_result.append(result)
            return np.array(all_result).T
    
    
    def show(self, labels=None, index=None, top=None, estimate=None):
        '''
        作图并显示
        
        参数
        ----
        labels：一维列表，可选，特征名称，默认为None
        index：一维列表，可选，特征索引，默认为None，表示全选
        top：整型，可选，表示显示值最高的前top个特征，默认为None，表示全选
        estimate：整型，可选，表示图示示数保留的小数位数，默认为None
        
        
        Display the image
        
        Parameters
        ----------
        labels: 1D list, callable, names of features, default=None
        index: 1D list, callable, index of features, default=None, which means select all
        top: int, callable, display "top" features with the highest values, default=None, which means select all
        estimate: int, callable, indicating the number of decimal places reserved for the graphic display, default=None
        '''
        _image_process(self.weight, labels, index, top, estimate)
        plt.show()
    
    
    def savefig(self, filename, labels=None, index=None, top=None, estimate=None):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串类型，文件名
        labels：一维列表，可选，特征名称，默认为None
        index：一维列表，可选，特征索引，默认为None，表示全选
        top：整型，可选，表示显示值最高的前top个特征，默认为None，表示全选
        estimate：整型，可选，表示图示示数保留的小数位数，默认为None
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        labels: 1D list, callable, names of features, default=None
        index: 1D list, callable, index of features, default=None, which means select all
        top: int, callable, display "top" features with the highest values, default=None, which means select all
        estimate: int, callable, indicating the number of decimal places reserved for the graphic display, default=None
        '''
        _image_process(self.weight, labels, index, top, estimate)
        plt.savefig(filename)