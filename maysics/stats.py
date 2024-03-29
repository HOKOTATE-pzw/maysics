'''
该模块用于统计分析

This module is uesd for statistical analysis
'''
import numpy as np
from matplotlib import pyplot as plt
from maysics.calc import inte
from maysics.utils import grid_net
from scipy.stats import chi2
from scipy.interpolate import interp1d


def _rc(arg):
    cov_mat = np.cov(arg)
    var_mat = np.diagonal(cov_mat)**0.5
    var_mat[var_mat == 0] = 1
    
    for i in range(cov_mat.shape[0]):
        cov_mat[i] /= var_mat[i]
        cov_mat[:, i] /= var_mat[i]
    
    return cov_mat


def r_moment(data, p_range=None, param={}, k=1, acc=0.1):
    '''
    计算原点矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    p_range：元组类型，可选，计算上下限，仅当data为函数类型时有效
    param：字典类型，可选，用于传递func中的其他参数，仅当data为函数类型时有效，默认为空字典
    k：整型，可选，阶数，默认为1
    acc：浮点数类型，可选，积分精度，仅在data为函数类型时有效，默认为0.1
    
    返回
    ----
    数，原点矩
    
    
    Calculate raw moment
    
    Parameters
    ----------
    fdata: 1D list or function, a sample or probability density function
    p_range：tuple, callable, lower and upper limit of calculation, valid only when data is a function
    param: dict, callable, pass other parameters to func, valid only when data is a function, default={}
    k: int, callable, orders, default=1
    acc: float, callable, integration accuracy, it's effective only when data is function, default=0.1
    
    Return
    ------
    num, raw moment
    '''
    if type(data).__name__ == 'function' or type(data).__name__ == 'method':
        def ex_func(x):
            return x**k * data(x, **param)
        
        return inte(ex_func, [p_range], acc=acc)
    
    else:
        data = np.array(data)**k
        return data.mean()


def ex(data, p_range=None, param={}, acc=0.1):
    '''
    计算数学期望，等价于一阶原点矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    p_range：元组类型，可选，计算上下限，仅当data为函数类型时有效
    param：字典类型，可选，用于传递func中的其他参数，仅当data为函数类型时有效，默认为空字典
    acc：浮点数类型，可选，积分精度，仅在data为函数类型时有效，默认为0.1
    
    返回
    ----
    数，数学期望
    
    
    Calculate mathematical expectation, which is equivalent to one order raw moment
    
    Parameters
    ----------
    fdata: 1D list or function, a sample or probability density function
    p_range：tuple, callable, lower and upper limit of calculation, valid only when data is a function
    param: dict, callable, pass other parameters to func, valid only when data is a function, default={}
    acc: float, callable, integration accuracy, it's effective only when data is function, default=0.1
    
    Return
    ------
    num, mathematical expectation
    '''
    if type(data).__name__ == 'function' or type(data).__name__ == 'method':
        def ex_func(x):
            return x * data(x, **param)
        
        return inte(ex_func, [p_range], acc=acc)
    
    else:
        data = np.array(data)
        return data.mean()


def c_moment(data, p_range=None, param={}, k=1, acc=0.1):
    '''
    计算中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    p_range：元组类型，可选，计算上下限，仅当data为函数类型时有效
    param：字典类型，可选，用于传递func中的其他参数，仅当data为函数类型时有效，默认为空字典
    k：整型，可选，阶数，默认为1
    acc：浮点数类型，可选，积分精度，仅在data为函数类型时有效，默认为0.1
    
    返回
    ----
    数，中心矩
    
    
    Calculate central moment
    
    Parameters
    ----------
    data: 1D list or function, a sample or probability density function
    p_range：tuple, callable, lower and upper limit of calculation, valid only when data is a function
    param: dict, callable, pass other parameters to func, valid only when data is a function, default={}
    k: int, callable, orders, default=1
    acc: float, callable, integration accuracy, it's effective only when data is function, default=0.1
    
    Return
    ------
    num, central moment
    '''
    raw_exp = ex(data, p_range, param)
    
    if type(data).__name__ == 'function' or type(data).__name__ == 'method':
        def ex_func(x):
            return (x-raw_exp)**k * data(x, **param)
        
        return inte(ex_func, [p_range], acc=acc)
    
    else:
        data = np.array(data) - raw_exp
        return r_moment(data, k=k)


def dx(data, p_range=None, param={}, acc=0.1):
    '''
    计算方差，等价于二阶中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    p_range：元组类型，可选，计算上下限，仅当data为函数类型时有效
    param：字典类型，可选，用于传递func中的其他参数，仅当data为函数类型时有效，默认为空字典
    acc：浮点数类型，可选，积分精度，仅在data为函数类型时有效，默认为0.1
    
    返回
    ----
    数，方差
    
    
    Calculate variance, which is equivalent to two orders central moment
    
    Parameters
    ----------
    data: 1D list or function, a sample or probability density function
    p_range：tuple, callable, lower and upper limit of calculation, valid only when data is a function
    param: dict, callable, pass other parameters to func, valid only when data is a function, default={}
    acc: float, callable, integration accuracy, it's effective only when data is function, default=0.1
    
    Return
    ------
    num, variance
    '''
    return c_moment(data, p_range, param, 2, acc)


def skew(data, p_range=None, param={}, acc=0.1):
    '''
    计算偏度，等价于三阶中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    p_range：元组类型，可选，计算上下限，仅当data为函数类型时有效
    param：字典类型，可选，用于传递func中的其他参数，仅当data为函数类型时有效，默认为空字典
    acc：浮点数类型，可选，积分精度，仅在data为函数类型时有效，默认为0.1
    
    返回
    ----
    数，偏度
    
    
    Calculate skewness, which is equivalent to three orders central moment
    
    Parameters
    ----------
    data: 1D list or function, a sample or probability density function
    p_range：tuple, callable, lower and upper limit of calculation, valid only when data is a function
    param: dict, callable, pass other parameters to func, valid only when data is a function, default={}
    acc: float, callable, integration accuracy, it's effective only when data is function, default=0.1
    
    Return
    ------
    num, skewness
    '''
    return c_moment(data, p_range, param, 3, acc)


def kurt(data, p_range=None, param={}, acc=0.1):
    '''
    计算峰度，等价于四阶中心矩
    
    参数
    ----
    data：一维列表或函数类型，样本或概率密度函数
    p_range：元组类型，可选，计算上下限，仅当data为函数类型时有效
    param：字典类型，可选，用于传递func中的其他参数，仅当data为函数类型时有效，默认为空字典
    acc：浮点数类型，可选，积分精度，仅在data为函数类型时有效，默认为0.1
    
    返回
    ----
    数，峰度
    
    
    Calculate kurtosis, which is equivalent to four orders central moment
    
    Parameters
    ----------
    data: 1D list or function, a sample or probability density function
    p_range：tuple, callable, lower and upper limit of calculation, valid only when data is a function
    param: dict, callable, pass other parameters to func, valid only when data is a function, default={}
    acc: float, callable, integration accuracy, it's effective only when data is function, default=0.1
    
    Return
    ------
    num, kurtosis
    '''
    return c_moment(data, p_range, param, 4, acc)


def mle(func, data, p_range, acc=0.1):
    '''
    最大似然法
    求解似然函数L(θ) = ln(∏ f(xi; θ)) = ∑ ln(f(xi; θ))在θ ∈ p_range范围内的最大值

    参数
    ----
    func：函数类型，密度概率函数，func的形式应如下：
        def func(x, param1, param2, ...):
        其中x需输入一个数或一个向量，是随机变量
    data：一维或二维数组，样本
    p_range：二维数组，func未确定参数的取值范围
    acc：数类型，可选，精度，默认为0.1
    
    返回
    ----
    1D ndarray，最佳的参数值


    Maximum Likelihood Estimate
    Find the maximum of likelihood function L(θ) = ln(∏ f(xi; θ)) = ∑ ln(f(xi; θ)) when θ ∈ p_range

    Parameters
    ----------
    func：function, probability density function, the form of func should be as follows：
        def func(x, param1, param2, ...):
        x is a number or a vector, random variable
    data：1D or 2D array，samples
    p_range：2D array, callable, value range of undetermined parameters
    acc：num, callable, accuracy, default=0.1
    
    Return
    ------
    1D ndarray, the best parameter value
    '''
    # 构建似然函数的相反数
    def Lmin(theta, data):
        result = 0
        for i in data:
            single_result = func(i, *theta)
            if single_result <= 0:
                return float('-inf')
            result += np.log(single_result)
        return result
    
    p_range = np.array(p_range)
    p_range_net = []
    for i in p_range:
        p_range_net.append(np.arange(i[0], i[1], acc))
    
    p_range_net = grid_net(*p_range_net)
    
    y = []
    for i in p_range_net:
        y.append(Lmin(i, data))
    y = np.array(y)
    x = np.argmax(y)
    
    return p_range_net[x]


def inde_test(data):
    '''
    独立性检验
    
    参数
    ----
    data：二维数组
    
    返回
    ----
    接受假设的错误概率，即认为两变量没有关系的错误概率
    
    
    Independence Test
    
    Parameter
    ---------
    data: 2D array
    
    Return
    ------
    the error probability of accepting the hypothesis (the error probability that the two variables are not related)
    '''
    data = np.array(data, dtype=float)
    data_shape = data.shape
    sum_1 = np.array([data.sum(axis=1)]).T
    sum_2 = data.sum(axis=0)
    p1 = data / sum_1
    p2 = data / sum_2
    sum_3 = data.sum()
    p_data = p1 * p2 * sum_3
    data = (data - p_data)**2 / p_data
    data = data.sum()
    
    return chi2.cdf(data, (data_shape[0] - 1) * (data_shape[1] - 1))


class DF1d():
    '''
    一维分布拟合
    用频率分布函数逼近概率密度函数
    
    参数
    ----
    sample：一维列表，样本点
    span：一维数组，区间间隔，如span = [a, b, c]则将区间分为[a, b]和[b, c]，并统计各区间频率
    kind：浮点数类型或整型，可选，将插值类型指定为字符串('linear'、'nearest'、'zero'、'slinear'、'squardic'、'previous'、'next'，其中'zero'、'slinear'、'squared'和'cubic'表示零阶、一阶、二阶或三阶样条曲线插值；'previous'和'next'只返回点的上一个或下一个值)或作为一个整数指定要使用的样条曲线插值器的顺序。默认为'linear'
    
    属性
    ----
    f：概率密度函数
    
    
    1D distribution fitting
    Approximation of probability density function by frequency distribution function
    
    Parameters
    ----------
    sample: 1D list, samples
    span: 1D array, span of intervals, e.g. span = [a, b, c] will divide the interval into [a, b] and [b, c], and count the frequency of each interval
    kind: float or int, callable, specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. default='linear'
    
    Attribution
    -----------
    f: probability density function
    '''
    def __init__(self, sample, span, kind='linear'):
        x_list = []
        y_list = []
        sample = np.array(sample)
        total_num = len(sample)
        for i in range(len(span)-1):
            num = len(sample[sample < span[i+1]]) - len(sample[sample < span[i]])
            y_list.append(num / total_num)
            x_list.append(0.5 * (span[i] + span[i+1]))
        self.f = interp1d(x_list, y_list, kind=kind)
        self.__minmax = [min(x_list), max(x_list)]
    
    
    def show(self, acc=0.01):
        '''
        作图并显示
        
        参数
        ----
        acc：浮点数类型，精度
        
        
        Display the image
        
        Parameters
        ----------
        acc: float, accuracy
        '''
        x = np.arange(self.__minmax[0], self.__minmax[1], acc)
        plt.plot(x, self.f(x))
        plt.show()
    
    
    def savefig(self, filename, acc=0.01):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串类型，保存的文件名
        acc：浮点数类型，精度
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        acc: float, accuracy
        '''
        x = np.arange(self.__minmax[0], self.__minmax[1], acc)
        plt.plot(x, self.f(x))
        plt.savefig(filename)


class DFT():
    '''
    单个分布拟合检验
    
    参数
    ----
    func_type：字符串类型，分布拟合采用的函数的类型，可选'pdf'(概率密度函数)、'cdf'(分布函数)和'pmf'(离散分布函数)，默认为'pdf'
    
    属性
    ----
    degree：卡方分布的自由度
    chi2_value：卡方值
    P：拒绝假设的错误概率
    
    
    Single distribution fitting test

    Parameter
    ---------
    func_type: str, callable, types of functions used in distribution fitting, 'pdf'(probability density function), 'cdf'(cumulative distribution function) and 'pmf'(discrete distribution function) are optional, default='pdf'
    
    Atrributes
    ----------
    degree: degrees of freedom of chi square distribution
    chi2_value: chi square value
    P: Probability of error in rejecting hypothesis
    '''
    def __init__(self, func_type='pdf'):
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
        
        return data_list, pro_list
    
    
    def __con_fit(self, data, func, num_data, param, acc):
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
                pro_list.append(inte(func, [[i[0], i[-1]]], param=param, acc=acc))
        
        elif self.__func_type == 'cdf':
            for i in data_list:
                pro_list.append(func(i[-1], **param) - func(i[0], **param))
        
        pro_list = np.array(pro_list) * num_data
        
        # 将数据数组转化为fi数组
        for i in range(len(data_list)):
            data_list[i] = len(data_list[i])
        data_list = np.array(data_list)
        data_list, pro_list = DFT.__merge(self, data_list, pro_list)
        return data_list, pro_list
    

    def __dis_fit(self, data, func, num_data, param):
        data = list(data)
        data_set = sorted(set(data))

        # 求出各data中个元素出现的次数并计算npi数组
        data_list = []
        pro_list = []
        for i in data_set:
            data_list.append(data.count(i))
            pro_list.append(func(i, **param))
        data_list = np.array(data_list)
        pro_list = np.array(pro_list) * num_data

        data_list, pro_list = DFT.__merge(self, data_list, pro_list)

        return data_list, pro_list
    
    
    def fit(self, data, func, param={}, acc=0.1):
        '''
        参数
        ----
        data：数据
        func：函数类型，分布函数或概率密度函数，函数的输入须为一个数
        param：字典类型，可选，用于传递func中的其他参数，默认为空字典
        acc：浮点数类型，可选，积分精度，仅func_type为'pdf'或'cdf'时有效，默认为0.1
        
        
        Parameters
        ----------
        data: data
        func: function, cumulative distribution function or probability density function, the input of func should be a number
        param: dict, callable, pass other parameters to func, default={}
        acc: float, callable, integration accuracy, it's effective only when func_type is 'pdf' or 'cdf', default=0.1
        '''
        self.__data = np.array(data, dtype=np.float)
        self.__func = func
        self.__param = param
        num_data = len(data)
        
        if self.__func_type != 'pmf':
            data_list, pro_list = self.__con_fit(data, func, num_data, param, acc)
        else:
            data_list, pro_list = self.__dis_fit(data, func, num_data, param)
        
        self.degree = len(pro_list) - 1
        self.chi2_value = sum((data_list - pro_list)**2 / pro_list)
        self.P = chi2.cdf(self.chi2_value, self.degree)
    
    
    def __image_process(self, acc):
        x_min = min(self.__data)
        x_max = max(self.__data)
        if self.__func_type == 'pmf':
            acc = 1
        x = np.arange(x_min, x_max, acc)
        y = self.__func(x, **self.__param)
        
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
        
        if self.__func_type != 'pmf':
            plt.plot(x, y)
        else:
            plt.scatter(x, y)
            zeros = np.zeros_like(y)
            plt.vlines(x, zeros, y)
        plt.yticks([])
    
    
    def show(self, acc=0.01):
        '''
        显示数据的小提琴图和假设的分布函数或概率密度图像
        
        参数
        ----
        acc：浮点数类型，可选，画图精度，仅func_type为'pdf'或'cdf'时有效，默认为0.01
        
        
        Display the violin chart of data and the image of hypothetical cumulative distribution function or probability density function
        
        Parameter
        ---------
        acc: float, callable, the accuracy of drawing, it's effective only when method is 'pdf' or 'cdf', default=0.01
        '''
        self.__image_process(acc)
        plt.show()
    
    
    def savefig(self, filename, acc=0.01):
        '''
        储存数据的小提琴图和假设的分布函数或概率密度图像
        
        参数
        ----
        filename：字符串类型，文件名
        acc：浮点数类型，可选，画图精度，仅在func_type为'pdf'或'cdf'时有效，默认为0.01
        
        
        Save the violin chart of data and the image of hypothetical cumulative distribution function or probability density function
        
        Parameters
        ----------
        filename: str, file name
        acc: float, callable, the accuracy of drawing, it's effective only when method is 'pdf' or 'cdf', default=0.01
        '''
        self.__image_process(acc)
        plt.savefig(filename)


class RC():
    '''
    相关系数
    
    参数
    ----
    *arg：列表类型
    
    属性
    ----
    rc_mat：相关系数矩阵
    
    
    correlation coefficient
    
    Parameter
    ---------
    *arg: list
    
    Attribute
    ---------
    rc_mat: correlation coefficient matrix
    '''
    def __init__(self, *arg):
        arg = np.array(arg, dtype=float)
        if len(arg.shape) != 2:
            raise Exception("Input list should be 1D.")
        else:
            self.rc_mat = _rc(arg)
    
    
    def __img_process(self, index, cmap):
        plt.matshow(self.rc_mat, cmap=cmap)
        plt.colorbar()
        if index:
            n_list = range(len(index))
            plt.xticks(n_list, index)
            plt.yticks(n_list, index)
    
    
    def show(self, index=None, cmap='Blues'):
        '''
        作图并显示
        
        参数
        ----
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Display the image
        
        Parameters
        ----------
        index: list, callable, names of each array
        cmap: str, callable, color board, default='Blues'
        '''
        self.__img_process(index=index, cmap=cmap)
        plt.show()
    
    
    def savefig(self, filename, index=None, cmap='Blues'):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串形式，文件名
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        index: list, callable, names of each array
        cmap: str, callable, color board, default='Blues'
        '''
        self.__img_process(index=index, cmap=cmap)
        plt.savefig(filename)


class PRC():
    '''
    偏相关系数
    等比例的数组不能用于该函数求解偏相关系数
    
    参数
    ----
    *arg：列表类型
    
    属性
    ----
    prc_mat：偏相关系数矩阵
    
    
    partial correlation coefficient
    Equally proportional arrays cannot be used to solve partial correlation coefficients for this function.
    
    Parameter
    ---------
    *arg: list
    
    Attribute
    ---------
    prc_mat: partial correlation coefficient matrix
    '''
    def __init__(self, *arg):
        arg = np.array(arg, dtype=float)
        if len(arg.shape) != 2:
            raise Exception("Input list should be 1D.")
        else:
            arg = np.linalg.inv(_rc(arg))
            self.prc_mat = np.zeros_like(arg)
            for i in range(self.prc_mat.shape[0]):
                for j in range(self.prc_mat.shape[1]):
                    self.prc_mat[i, j] = - arg[i, j] / (arg[i, i] * arg[j, j])**0.5
    
    
    def __img_process(self, index, cmap):
        plt.matshow(self.prc_mat, cmap=cmap)
        plt.colorbar()
        if index:
            n_list = range(len(index))
            plt.xticks(n_list, index)
            plt.yticks(n_list, index)
    
    
    def show(self, index=None, cmap='Blues'):
        '''
        作图并显示
        
        参数
        ----
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Display the image
        
        Parameters
        ----------
        index: list, callable, names of each array
        cmap: str, callable, color board, default='Blues'
        '''
        self.__img_process(index=index, cmap=cmap)
        plt.show()
    
    
    def savefig(self, filename, index=None, cmap='Blues'):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串形式，文件名
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        index: list, callable, names of each array
        cmap: str, callable, color board, default='Blues'
        '''
        self.__img_process(index=index, cmap=cmap)
        plt.savefig(filename)