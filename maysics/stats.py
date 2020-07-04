'''
该模块用于统计分析

This module is uesd for statistical analysis
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.optimize import minimize



def r_moment(data, a=None, b=None, k=1):
    '''
    计算原点矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    a：浮点数类型，可选，计算下限
    b：浮点数类型，可选，计算上限
    k：整型，可选，阶数
    
    
    Calculate raw moment
    
    Parameters
    ----------
    fdata: 1-D list or function, a sample or probability density function
    a: float, callable, lower limit of calculation
    b: float, callable, upper limit of calculation
    k: int, callable, orders
    '''
    if type(data).__name__ == 'function':
        def ex_func(x):
            return x**k * data(x)
        
        return quad(ex_func, a, b)[0]
    
    else:
        data = np.array(data)**k
        return data.mean()


def EX(data, a=None, b=None):
    '''
    计算数学期望，等价于一阶原点矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    a：浮点数类型，可选，计算下限
    b：浮点数类型，可选，计算上限
    
    
    Calculate mathematical expectation, which is equivalent to one order raw moment
    
    Parameters
    ----------
    fdata: 1-D list or function, a sample or probability density function
    a: float, callable, lower limit of calculation
    b: float, callable, upper limit of calculation
    '''
    if type(data).__name__ == 'function':
        def ex_func(x):
            return x * data(x)
        
        return quad(ex_func, a, b)[0]
    
    else:
        data = np.array(data)
        return data.mean()


def c_moment(data, k, a=None, b=None):
    '''
    计算中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    a：浮点数类型，可选，计算下限
    b：浮点数类型，可选，计算上限
    k：整型，可选，阶数
    
    
    Calculate central moment
    
    Parameters
    ----------
    data: 1-D list or function, a sample or probability density function
    a: float, callable, lower limit of calculation
    b: float, callable, upper limit of calculation
    k: int, callable, orders
    '''
    raw_exp = EX(data, a, b)
    
    if type(data).__name__ == 'function':
        def ex_func(x):
            return (x-raw_exp)**k * data(x)
        
        return quad(ex_func, a, b)[0]
    
    else:
        data = np.array(data) - raw_exp
        return r_moment(data, k=k)


def DX(data, a=None, b=None):
    '''
    计算方差，等价于二阶中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    a：浮点数类型，可选，计算下限
    b：浮点数类型，可选，计算上限
    
    
    Calculate variance, which is equivalent to two orders central moment
    
    Parameters
    ----------
    data: 1-D list or function, a sample or probability density function
    a: float, callable, lower limit of calculation
    b: float, callable, upper limit of calculation
    '''
    return c_moment(data, 2, a, b)


def skew(data, a=None, b=None):
    '''
    计算偏度，等价于三阶中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    a：浮点数类型，可选，计算下限
    b：浮点数类型，可选，计算上限
    
    
    Calculate skewness, which is equivalent to three orders central moment
    
    Parameters
    ----------
    data: 1-D list or function, a sample or probability density function
    a: float, callable, lower limit of calculation
    b: float, callable, upper limit of calculation
    '''
    return c_moment(data, 3, a, b)


def kurt(data, a=None, b=None):
    '''
    计算峰度，等价于四阶中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    a：浮点数类型，可选，计算下限
    b：浮点数类型，可选，计算上限
    
    
    Calculate kurtosis, which is equivalent to four orders central moment
    
    Parameters
    ----------
    data: 1-D list or function, a sample or probability density function
    a: float, callable, lower limit of calculation
    b: float, callable, upper limit of calculation
    '''
    return c_moment(data, 4, a, b)


def mle(func, data, num, p_range=(-1, 1), method=None, tol=None):
    '''
    最大似然法
    求解似然函数L(θ) = ln(∏ f(xi; θ)) = ∑ ln(f(xi; θ))在θ ∈ p_range范围内的最大值

    参数
    ----
    func：函数类型，密度概率函数，func的形式应如下：
        def func(x, param):
        其中x需输入一个数或一个向量，是随机变量；param是一维ndarray，为未确定的参数
    data：一维或二维列表，样本
    num：未确定参数个数
    p_range：一维或二维列表，未确定参数的取值范围，若为一维列表，则代表所有未确定参数的取值范围都一样，默认为(-1, 1)
    method：字符串类型，求最大值的方法，可选'Nelder-Mead'、'Powell'、'CG'、'BFGS'、'Newton-CG'、'L-BFGS-B'、'TNC'、'COBYLA'、'SLSQP'、'trust-constr'、'dogleg'、'trust-ncg'、'trust-exact'、'trust-krylov'
    tol：误差


    Maximum Likelihood Estimate
    Find the maximum of likelihood function L(θ) = ln(∏ f(xi; θ)) = ∑ ln(f(xi; θ)) when θ ∈ p_range

    Parameters
    ----------
    func：function, probability density function, the form of func should be as follows：
        def func(x, param):
        x is a number or a vector, random variable; param is 1-D ndarray, undetermined parameters
    data：1-D or 2-D list，samples
    num：the number of undetermined parameters
    p_range：1-D or 2-D list, value range of undetermined parameters, in case of one-dimensional list，all the undetermined parameters are in the same range, default=(-1, 1)
    method：str，the method of finding the maximum, 'Nelder-Mead'、'Powell'、'CG'、'BFGS'、'Newton-CG'、'L-BFGS-B'、'TNC'、'COBYLA'、'SLSQP'、'trust-constr'、'dogleg'、'trust-ncg'、'trust-exact'、'trust-krylov' are optional
    tol：error
    '''
    # 构建似然函数的相反数
    def Lmin(theta):
        result = 0
        for i in data:
            result += np.log(func(i, theta))
        return result * (-1)

    x0 = []
    constraints = []
    p_range = np.array(p_range)
    if len(p_range.shape) == 1:
        p_range = np.tile(p_range, (num, 1))
    for i in range(num):
        x0.append(p_range[i][0])
        constraints.append({'type':'ineq', 'fun':lambda x: x[i] - p_range[i][0]})
        constraints.append({'type':'ineq', 'fun':lambda x: p_range[i][1] - x[i]})
    x0 = np.array(x0)
        
    res = minimize(Lmin, x0, method=method, constraints=constraints, tol=tol)
        
    return res.x



class DFT():
    '''
    单个分布拟合检验
    
    参数
    ----
    func_type：字符串类型，分布拟合采用的函数的类型，可选'pdf'(概率密度函数)、'cdf'(分布函数)和'dis'(离散分布函数)，默认为'pdf'
    
    属性
    ----
    degree：卡方分布的自由度
    chi2_value：卡方值
    P：拒绝假设的错误概率
    
    
    Single distribution fitting test

    Parameter
    ---------
    func_type: str, callable, types of functions used in distribution fitting, 'pdf'(probability density function), 'cdf'(cumulative distribution function) and 'dis'(discrete distribution function) are optional, default='pdf'
    
    Atrributes
    ----------
    degree: degrees of freedom of chi square distribution
    chi2_value: chi square value
    P: Probability of error in rejecting hypothesis
    '''
    def __init__(self, func_type=None):
        if not func_type:
            self.__func_type = 'pdf'
        else:
            self.__func_type = func_type
    
    
    def __merge(self, data_list, pro_list):
        # 找到npi < 5的元素位置
        delete_index = np.where(pro_list < 5)[0]
        if len(delete_index) > 0:
            delete_pro = pro_list[delete_index]
            delete_data = data_list[delete_index]
            pro_final = [delete_pro[0]]
            data_final = [delete_data[0]]
            for i in range(1, len(delete_pro)):
                if pro_final[-1] < 5:
                    pro_final[-1] += delete_pro[i]
                    data_final[-1] += delete_data[i]
                else:
                    pro_final.append(delete_pro[i])
                    data_final.append(delete_data[i])
            pro_final = np.array(pro_final)
            data_final = np.array(data_final)
            
            # 删除原列表npi < 5位置对应的元素
            data_list = np.delete(data_list, delete_index)
            pro_list = np.delete(pro_list, delete_index)
            
            data_list = np.hstack((data_list, data_final))
            pro_list = np.hstack((pro_list, pro_final))
            
            if len(pro_list) == 1:
                raise Exception("Degree of freedom of chi-square distribution is 0.")
            
            elif len(pro_list) > 1 and pro_list[-1] < 5:
                data_list[-2] += data_list[-1]
                pro_list[-2] += pro_list[-1]
                data_list = np.delete(data_list, -1)
                pro_list = np.delete(pro_list, -1)
            
        # 计算fi^2数组
        data_list = data_list**2
        
        return data_list, pro_list
    
    
    def __con_fit(self, data, func, num_data):
        data = sorted(data)
        
        # 从小到大按20个一组分成多个部分
        data_list = []
        for i in range(int(num_data / 20 + 1)):
            data_list.append(data[i*20: (i+1)*20])
        
        # 如果最后一个部分的数量只有0个或1个元素，则叠加到前面
        if len(data_list[-1]) == 1:
            data_list[-2].append(data_list[-1][0])
            del data_list[-1]
        
        elif len(data_list[-1]) == 0:
            del data_list[-1]
        
        # 计算npi数组
        pro_list = []
        if self.__func_type == 'pdf':
            for i in data_list:
                pro_list.append(quad(func, i[0], i[-1])[0])
        
        elif self.__func_type == 'cdf':
            for i in data_list:
                pro_list.append(func(i[-1]) - func(i[0]))
        
        pro_list = np.array(pro_list) * num_data
        
        # 将数据数组转化为fi数组
        for i in range(len(data_list)):
            data_list[i] = len(data_list[i])
        data_list = np.array(data_list)
        
        data_list, pro_list = DFT.__merge(self, data_list, pro_list)

        return data_list, pro_list
    

    def __dis_fit(self, data, func, num_data):
        data = list(data)
        data_set = sorted(set(data))

        # 求出各data中个元素出现的次数并计算npi数组
        data_list = []
        pro_list = []
        for i in data_set:
            data_list.append(data.count(i))
            pro_list.append(func(i))
        data_list = np.array(data_list)
        pro_list = np.array(pro_list) * num_data

        data_list, pro_list = DFT.__merge(self, data_list, pro_list)

        return data_list, pro_list
    
    
    def fit(self, data, func):
        '''
        参数
        ----
        data：数据
        func：函数类型，分布函数或概率密度函数，函数的输入须为一个数
        
        
        Parameters
        ----------
        data: data
        func: function, cumulative distribution function or probability density function, the input of func should be a number
        '''
        self.__data = np.array(data, dtype=np.float)
        self.__func = func
        num_data = len(data)
        
        if self.__func_type != 'dis':
            data_list, pro_list = DFT.__con_fit(self, data, func, num_data)
        else:
            data_list, pro_list = DFT.__dis_fit(self, data, func, num_data)
        
        self.degree = len(pro_list) - 1
        self.chi2_value = sum(data_list / pro_list) - num_data
        self.P = 1 - chi2.cdf(self.chi2_value, self.degree)
    
    
    def __image_process(self, acc):
        x_min = min(self.__data)
        x_max = max(self.__data)
        x = np.arange(x_min, x_max, acc)
        y = self.__func(x)
        
        func_type = type(y).__name__
        if func_type == 'int' or func_type == 'float':
            y = np.ones_like(x) * 0.3
        else:
            y *= (0.3 / y.max())
        
        quartile1, medians, quartile3 = np.percentile(self.__data, [25, 50, 75])
        
        ax = plt.violinplot(self.__data, widths=0.6, showextrema=False, vert=False)
        ax['bodies'][0].set_facecolor('red')
        ax['bodies'][0].set_edgecolor('black')
        ax['bodies'][0].set_alpha(1)
        
        plt.scatter(medians, 1, marker='o', color='white', s=30, zorder=3)
        plt.hlines(1, quartile1, quartile3, color='k', linestyle='-', lw=5)
        plt.hlines(1, x_min, x_max, color='k', linestyle='-', lw=1)
        plt.vlines(self.__data.mean(), 0.9, 1.1, color='yellow', linestyle='-', lw=2)
        
        plt.plot(x, y)
        plt.yticks([])
    
    
    def show(self, acc=0.01):
        '''
        显示数据的小提琴图和假设的分布函数或概率密度图像
        
        参数
        ----
        acc：浮点数类型，可选，画图精度，默认为0.01
        
        
        Display the violin chart of data and the image of hypothetical cumulative distribution function or probability density function
        
        Parameter
        ---------
        acc: float, callable, the accuracy of drawing, default=0.01
        '''
        DFT.__image_process(self, acc)
        plt.show()
    
    def savefig(self,filename, acc=0.01):
        '''
        储存数据的小提琴图和假设的分布函数或概率密度图像
        
        参数
        ----
        filename：字符串类型，文件名
        acc：浮点数类型，可选，画图精度，默认为0.01
        
        
        Save the violin chart of data and the image of hypothetical cumulative distribution function or probability density function
        
        Parameters
        ----------
        filename: str, file name
        acc: float, callable, the accuracy of drawing, default=0.01
        '''
        DFT.__image_process(self, acc)
        plt.savefig(filename)