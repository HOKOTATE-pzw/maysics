'''
本模块储存着部分常用的模型、算法

This module stores some commonly used models, algorithmns
'''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import random
from multiprocessing import Pool, cpu_count
from maysics.calc import grad


def linear_r(x, y):
    '''
    线性回归模型
    
    参数
    ----
    x：一维或二维ndarray，自变量
    y：一维或二维ndarray，因变量
    
    返回
    ----
    一维ndarray：系数行矩阵
    损失函数：平均平方误差mse
    
    
    Linear Regression Model
    
    Parameters
    ----------
    x: 1D or 2D ndarray, independent variable
    y: 1D or 2D ndarray, dependent variable
    
    Return
    ------
    1D ndarray: coefficient row matrix
    loss: mean squared error: mse
    '''
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.array([x]).T
    y = np.array(y)
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    coef = np.linalg.lstsq(x, y, rcond=None)[0].reshape(-1)
    y_pre = np.dot(x, np.array([coef]).T).reshape(y.shape)
    mse = ((y_pre - y)**2).mean()
    return coef, mse


def logistic(t, N0, r, K):
    '''
    Logistic人口增长模型
    该模型可得到解析解
    解的形式为：Nt = K * N0 / (N0 + (K - N0) * e**(-r * t))
    其中，Nt为t时刻的人口数
    
    参数
    ----
    t：时间
    N0：数或数组，现有人口数
    r：数类型，人口自然增长率
    K：数类型，环境资源允许的稳定人口数
    
    返回
    ----
    数，Nt
    
    
    Logistic population growth models
    solution: Nt = K * N0 / (N0 + (K - N0) * e**(-r * t))
    Nt is the population at time 't'
    
    Parameters
    ----------
    t: time
    N0: num or array, initial population
    r: num, natural population growth rate
    K: num, stable population allowed by environmental resources
    
    Return
    ------
    num, Nt
    '''
    Nt_pre_down_ = N0 + (K - N0) * np.e**(-r * t)
    Nt = K * N0 / Nt_pre_down_
    return Nt


def pagerank(data, loop=5, pr=None, d=0.85, l=False):
    '''
    网页排序算法
    
    参数
    ----
    data：列表形式，每个连接所指向的链接或L矩阵，L矩阵即L(i, j)表示：如果j链接指向i链接，则L(i, j)为j链接指向的所有链接数；否则为0
    loop：整型，可选，迭代次数，默认为5
    pr：一维数组形式，可选，初始的pagerank值，默认pagerank值全部相等
    d：数类型，可选，系数，默认为0.85
    l：布尔类型，可选，True表示data是L矩阵，默认为False
    
    返回
    ----
    一维ndarray，pagerank值，归一化为1
    
    
    Page Rank
    
    Parameters
    ----------
    data: list, the link to which each connection points or, matrix L, whose L(i, j)means: if j link points to i link, L(i, j) is the sum of all links pointed to by j link; otherwise, 0
    loop: int, callable, the number of iteration, default = 5
    pr: 1D array, callable, the original pagerank value, by default, all the values are equal
    d: num, callable, coeficient, default = 0.85
    l: bool, callable, True means data is matrix L, default = False
    
    Return
    ------
    1D ndarray, pagerank values, normalized to 1
    '''
    n_page = len(data)
    if pr is None:
        pr_list = np.zeros((n_page))
    else:
        pr_list = np.array(pr).copy()
    
    if l is False:
        L = np.zeros((n_page, n_page))
        for i in range(n_page):
            for j in range(n_page):
                if i in data[j]:
                    L[i, j] = len(data[j])
        L[L!=0] = 1 / L[L!=0]
    else:
        L = np.array(data)
    
    for i in range(loop):
        pr_list = np.dot(pr_list, L.T)
        if pr_list.sum() != 0:
            pr_list /= pr_list.sum()
        pr_list = pr_list * d + (1 - d) / n_page
    return pr_list


def pso(select, initial, num=10, loop=10, omega=1, phi_1=2, phi_2=2, v_max=None, param={}, random_state=None, batch=True):
    '''
    粒子群优化算法
    
    参数
    ----
    select：函数，粒子的评估函数，需返回每个粒子的评估值，默认函数最小值为最优
    initial：1维或2维数组，初始粒子位置 
    num：整型，可选，模拟粒子个数，默认为10
    loop：整型，可选，迭代次数，默认为10
    omega：数或函数，可选，惯性权重因子，若为函数，其输入须为迭代次数，默认为1
    phi_1：数类型，可选，第一个加速度常数，默认为2
    phi_2：数类型，可选，第二个加速度常数，默认为2
    v_max：数类型，可选，粒子最大速度
    param：字典类型，可选，当select函数有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    random_state：整型，可选，随机种子
    batch：布尔类型，True表示评估函数为批量输入函数，False表示评估函数为非批量输入函数，默认为True
    
    返回
    ----
    一维ndarray，最优粒子位置
    
    
    Particle Swarm Optimization
    
    Parameters
    ----------
    select: function, estimation function
    initial: 1D or 2D array, the initial location of particles
    num: int, callable, the number of particles, default=10
    loop: int, callable, the number of iteration, default=10
    omega: num or function, callable, inertia weight factor, if it is function, it should take the number of iteration as input, default=1
    phi_1: num, callable, the first acceleration constant, default=2
    phi_2: num, callable, the second acceleration constant, default=2
    v_max: num, callable, the maximal velocity of particles
    param: dict, callable, when function "select" has other non-default parameters, "param" needs to be input as a dictionary with parm_name as key and param_value as value, default={}
    random_state: int, callable, random seed
    batch: bool, callable, True means estimation function is batch-input, False means not
    
    Return
    ------
    1D ndarray, location of the optimized particle
    '''
    np.random.seed(random_state)
    initial = np.array(initial, dtype=float)
    initial = np.tile(initial, (num, 1))
    v = np.ones_like(initial)
    p = initial.copy()
    p_g = p[0].copy()
    
    if batch is True:
        values = list(select(initial, **param))
    
    else:
        values = []
        for i in initial:
            values.append(select(i, **param))
    
    for i in range(loop):
        initial_shape = initial.shape
        if not type(omega).__name__ == 'function':
            v = omega * v + phi_1 * np.random.rand(*initial_shape) * (p - initial)\
                          + phi_2 * np.random.rand(*initial_shape) * (p_g - initial)
        else:
            v = omega(i) * v + phi_1 * np.random.rand(*initial_shape) * (p - initial)\
                             + phi_2 * np.random.rand(*initial_shape) * (p_g - initial)
        
        if not v_max is None:
            v[v > v_max] = v_max
            v[v < -v_max] = -v_max
        initial += v
        
        if batch is True:
            values_new = select(initial, **param)
            loc = np.where(np.array(values_new) < np.array(values))[0]
            p[loc] = initial[loc]
        
        else:
            values_new = []
            for i in range(initial_shape[0]):
                value_new = select(initial[i], **param)
                values_new.append(value_new)
                if value_new < values[i]:
                    p[i] = initial[i]
        
        values = values_new
        index = np.argmin(values)
        p_g = p[index]
        
    return p_g


def simple_gd(select, initial, ytol, acc, step=-7.0, auto=True, param={}):
    '''
    简化版梯度下降算法
    
    参数
    ----
    select：函数类型，评估函数
    initial：数或数组，初始解，select函数的输入值
    ytol：浮点数类型，可选，连续两次迭代的函数值小于ytol时即停止迭代，默认为0.01
    acc：浮点数类型，可选，求导精度，默认为0.1
    step：浮点数类型，可选，步长倍率，每次生成的步长为step * 负梯度，若auto=True，则步长为x*10^step，其中x为梯度的第一位有效数字，默认为0.1
    auto：布尔类型，True使用自适应步长，默认为True
    param：字典类型，可选，当select有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    
    返回
    ----
    ndarray，最优解
    
    
    Simple Gradient Descent
    
    Parameters
    ----------
    select: function, evaluation function
    initial: num or array, initial solution, the input value of select function
    ytol:
    acc:
    step:
    auto:
    param: dict, callable, when function "select" has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    
    Return
    ------
    ndarray, optimized solution
    '''
    initial = np.array(initial, dtype=float)
    f_change = float('inf')
    
    while f_change > ytol:
        d_list = grad(select, initial, acc, param=param)
        if auto:
            step = -1 * (np.floor(np.log(d_list) / np.log(10)).astype(float)) + step
            step = 10**step
        # 计算函数值变化量
        f_change = select(initial, **param)
        initial = initial - d_list * step
        f_change = abs(select(initial, **param) - f_change)
    
    return initial


class AR():
    '''
    自回归模型
    X(t) = (φt-1, φt-2, ..., φt-p) · (X(t-1), X(t-2), ..., X(t-p)).T + c + εt
    其中，φt-i是系数，c是常数，εt是随机变量
    
    属性
    ----
    coef：1维ndarray，系数数组
    sigma：数类型，原数据与预测之间误差的方差
    
    
    Autoregressive Model
    X(t) = (φt-1, φt-2, ..., φt-p) · (X(t-1), X(t-2), ..., X(t-p)).T + c + εt
    where, φt-i is coefficient, c is constant, εt is a random variable
    
    Attributes
    ----------
    coef: 1D array, coefficient array
    sigma: num, variance of error between original data and prediction
    '''
    @classmethod
    def fit(self, data, p=1):
        '''
        参数
        ----
        data：数组，需要回归的数据
        p：整型，可选，选择进行回归分析的历史项数，默认为1
        
        
        Paramaeters
        -----------
        data: array, data to be regressed
        p: int, callable, number of historical items selected for regression analysis, default=1
        '''
        self.__p = p
        data = np.array(data)
        if len(data.shape) == 2:
            self.__data = data.T[0]
        else:
            self.__data = data
        X = np.ones((self.__data.shape[0]-p, p+1))
        for i in range(X.shape[0]):
            X[i, :-1] = self.__data[i:p+i][::-1]
        y = self.__data[p:]
        self.coef = np.linalg.lstsq(X, y, rcond=-1)[0]
        
        self.sigma = []
        for i in range(self.__data[p:].shape[0]):
            self.sigma.append(self.coef[:p] * self.__data[i:i+p] - self.__data[i+p])
        self.sigma = np.array(self.sigma).std()
    
    
    @classmethod
    def predict(self, t, sigma=None, recount=False, mean=False, random_state=None, method='norm', param={}):
        '''
        参数
        ----
        t：数或数组类型，预测的时间
        sigma：数类型，可选，随机变量的方差，默认与误差方差相同
        recount：布尔类型，可选，True代表预测时间以原数据的最后一个时间点为-1重新计算，默认为Fasle
        mean：布尔类型，可选，True代表以预测均值输出，默认为False
        random_state：整型，可选，随机种子
        method：字符串类型或函数类型，字符串可选'norm'和'uniform'，分别代表正态分布和均匀分布，默认为'norm'
        param：字典类型，可选，当method为函数且有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        
        返回
        ----
        1维ndarray
        
        
        Parameters
        ----------
        t: num or array, time
        sigma: num, callable, the variance of random variables, which is the same as the error variance by default
        recount: bool, callable, True means the prediction time is recalculated with the last time point of the original data as - 1, default=False
        mean: bool, callabel, True means output with the predicted mean value, default=False
        random_state: int, callable, random seed
        method: str or function, callable, the string can be 'norm' or 'uniform', representing normal distribution and uniform distribution respectively, default='norm'
        param: dict, callable, when "mehod" is function and has other non-default parameters, "param" needs to be input as a dictionary with parm_name as key and param_value as value, default={}
        
        Return
        ------
        1D ndarray
        '''
        if not sigma:
            sigma = self.sigma
        np.random.seed(random_state)
        
        if type(t) == list or type(t) == tuple:
            t = np.array(t)
        elif type(t) == int or type(t) == float:
            t = np.array([t])
        
        if recount is False:
            t -= self.__data.shape[0]
        
        t = t.astype(int)
        
        output_data = self.__data[-self.__p:].copy()
        for i in range(t.max()+1):
            output_data = np.append(output_data,
                                    (output_data[-self.__p:] * self.coef[:self.__p]).sum() + self.coef[-1])
        
        output_data = output_data[self.__p:]
        output_data = output_data[t]
        
        if mean is False:
            if method == 'norm':
                output_data += np.random.normal(output_data, sigma, output_data.shape)
            elif method == 'uniform':
                output_data += np.random.uniform(output_data - 3**0.5 * sigma,
                                                 output_data + 3**0.5 * sigma,
                                                 output_data.shape)
            elif type(method).__name__ == 'function':
                output_data = method(output_data, **param)
        
        return output_data


class ED():
    '''
    传染病模型
    本类可构建SI模型、SIR模型、SIRS模型、SEIR模型
    S是易感染者
    E是暴露者，染病但还不具有传染能力
    I是感染者，带有传染能力
    R是康复者(带有免疫力)或死者
    
    参数
    ----
    I0：数类型，初始感染者
    K：数类型，样本总数
    beta：数类型，感染率
    R：数类型，可选，初始康复者(带有免疫力)或死者，默认为0
    E：数类型，可选，暴露者，默认为0
    
    
    mathematical models of epidemic diseases
    this class is used to buil SI, SIR, SIRS, SEIR
    S means susceptible people
    E means exposed people, which are infected without ability to infect
    I means infected people with ability to infect
    R means recovered people with immunity or dead people
    
    Parameters
    ----------
    I0: initial infected people
    K: sample size
    beta: infection rate
    R: num, callable, recovered people with immunity or dead people, default=0
    E: num, callable, exposed people, default=0
    '''
    def __init__(self, I0, K, beta, R=0, E=0):
        self.I0 = I0
        self.K = K
        self.beta = beta
        self.R = R
        self.E = E
        self.S = K - I0 - R - E
    
    
    def SI(self, t):
        '''
        SI模型：
        该模型不需要再额外输入参数，且可得到解析解
        解的形式为：I = K*(1-(K-I0)/(K+I0+I0*e**(beta*K*t)))
        
        参数
        ----
        t：数或一维数组，需要预测的时间

        返回
        ----
        元组，(S(t), I(t))
        
        
        SI:
        no more parameters
        solution: I = K*(1-(K-I0)/(K+I0+I0*np.e**(beta*K*t)))
        
        Parameter
        ---------
        t: num or 1D array, time for prediction
        
        Return
        ------
        tuple, (S(t), I(t))
        '''
        I_pre_down_ = self.K + self.I0 * (np.e**(self.beta * self.K * t) - 1)
        I_pre_up_ = self.K - self.I0
        I_func_ = self.K * (1 - I_pre_up_ / I_pre_down_)
        S_func_ = self.K - I_func_
        return S_func_, I_func_
    
    
    def SIR(self, gamma, t_span, method='RK45', t_eval=None):
        '''
        SIR模型：
        该模型不能得到解析解，给出的是基于solve_ivp得到的数值解数组
        
        参数
        ----
        gamma：数类型，康复率
        t_span：元组类型，求解的上下限
        method：字符串类型，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组类型，可选，每当t等于该数组中的值时，会生成一个数值解

        返回
        ----
        solve_ivp数值解，顺序是S, I, R
        
        
        SIR:
        
        Parameters
        ----------
        gamma: num, recovery rate
        t_span: tuple, limits of solution
        method: str, callable, solving method, 'RK45', 'RK23', 'DOP835', 'Radau', 'BDF', 'LSODA' are optional
        t_eval: list, callable, when 't' equals the value in the list, it will generate a numerical solution
        
        Return
        ------
        numerical solution of solve_ivp, the order is S, I, R
        '''
        y0 = [self.S, self.I0, self.R]
        def epis_equas(t, x):
            y1 = -self.beta * x[1] * x[0]
            y2 = self.beta * x[1] * x[0] - gamma * x[1]
            y3 = gamma * x[1]
            return y1, y2, y3
        result = solve_ivp(epis_equas, t_span=t_span, y0=y0, method=method, t_eval=t_eval)
        return result.y
    
    
    def SIRS(self, gamma, alpha, t_span, method='RK45', t_eval=None):
        '''
        SIRS模型：
        该模型不能得到解析解，给出的是基于solve_ivp得到的数值解数组
        
        参数
        ----
        gamma：数类型，康复率
        alpha：数类型，衡量康复者获得免疫的时间
        t_span：元组类型，求解的上下限
        method：字符串类型，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组类型，可选，每当t等于该数组中的值时，会生成一个数值解

        返回
        ----
        solve_ivp数值解，顺序是S, I, R
        
        
        SIRS:
        
        Parameters
        ----------
        gamma: num, recovery rate
        alpha: num, measuring the time that the recovered people with immunity
        t_span: tuple, limits of solution
        method: str, callable, solving method, 'RK45', 'RK23', 'DOP835', 'Radau', 'BDF', 'LSODA' are optional
        t_eval: list, callable, when 't' equals the value in the list, it will generate a numerical solution
        
        Return
        ------
        numerical solution of solve_ivp, the order is S, I, R
        '''
        y0= [self.S, self.I0, self.R]
        def epis_equas(t, x):
            y1 = -self.beta * x[1] * x[0] + alpha * x[2]
            y2 = self.beta * x[1] * x[0] - gamma * x[1]
            y3 = gamma * x[1] - alpha * x[2]
            return y1, y2, y3
        result = solve_ivp(epis_equas, t_span=t_span, y0=y0, method=method, t_eval=t_eval)
        return result.y
    
    
    def SEIR(self, gamma1, gamma2, alpha, t_span, method='RK45', t_eval=None):
        '''
        SEIR模型：
        该模型不能得到解析解，给出的是基于solve_ivp得到的数值解数组
        
        参数
        ----
        gamma1：数类型，潜伏期康复率
        gamma2：数类型，患者康复率
        alpha：数类型，衡量康复者获得免疫的时间
        t_span：元组类型，求解的上下限
        method：字符串类型，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组类型，可选，每当t等于该数组中的值时，会生成一个数值解

        返回
        ----
        solve_ivp数值解，顺序是S, E, I, R
        
        
        SEIR:
        
        Parameters
        ----------
        gamma1: num, recovery rate of incubation
        gamma2: num, recovery rate of patients
        alpha: num, measuring the time that the recovered people with immunity
        t_span: tuple, limits of solution
        method: str, callable, solving method, 'RK45', 'RK23', 'DOP835', 'Radau', 'BDF', 'LSODA' are optional
        t_eval: list, callable, when 't' equals the value in the list, it will generate a numerical solution
        
        Return
        ------
        numerical solution of solve_ivp, the order is S, E, I, R
        '''
        y0 = [self.S, self.E, self.I0, self.R]
        def epis_equas(t, x):
            y1 = -self.beta * x[2] * x[0]
            y2 = self.beta * x[2] * x[0] - (alpha + gamma1) * x[1]
            y3 = alpha * x[1] - gamma2 * x[2]
            y4 = gamma1 * x[1] + gamma2 * x[2]
            return y1, y2, y3, y4
        result = solve_ivp(epis_equas, t_span=t_span, y0=y0, method=method, t_eval=t_eval)
        return result.y


class GM():
    '''
    灰色系统模型，GM(1, 1)模型
    fit函数输入一维数组y：[y1, y2, ..., yn]
    对应的时间数组t为：[1, 2, ..., n]
    预测式：
        x(1)(t) = [x(0)(1) - b / a] * e**(- a * (t - 1)) + b / a  (t ∈ N+)
        t >= 2时：x(0)(t) = x(1)(t) - x(1)(t - 1)
        t == 1时：x(0)(t) = x(1)(t)
    
    属性
    ----
    C：数类型，调整级比范围时y数组的平移量（y + C）
    u：二维ndarray，列矩阵[a b].T
    
    
    Grey Model, GM(1, 1)
    The fit function inputs 1-D array y: [y1, y2, ..., yn]
    The corresponding time array t: [1, 2, ..., n]
    Prediction function:
        x(1)(t) = [x(0)(1) - b / a] * e**(- a * (t - 1)) + b / a  (t ∈ N+)
        t >= 2时：x(0)(t) = x(1)(t) - x(1)(t - 1)
        t == 1时：x(0)(t) = x(1)(t)
    
    Attributes
    ----------
    C: num, the translation of Y array when adjusting the range of stage ratio(y + C)
    u: 2-D ndarray, column matrix [a b].T
    '''
    @classmethod
    def fit(self, y, acc=1):
        '''
        进行GM(1, 1)拟合
        
        参数
        ----
        y：一维数组
        acc：数类型，可选，调整级比的精度，默认为1
        
        
        Fit with GM(1, 1)
        
        Parameter
        ---------
        y: 1-D array
        acc: num, callable, accuracy of adjusting stage ratio, default=1
        '''
        # 调整级比范围
        self.__y = np.array(y, dtype=float)
        n = len(self.__y)
        y_k_1 = self.__y[:-1]
        y_k = self.__y[1:]
        l_k = y_k_1 / y_k
        theta_min = np.e**(-2 / (n + 1))
        theta_max = np.e**(2 / (n + 2))

        self.C = 0
        while True:
            if np.min(l_k) <= theta_min or np.max(l_k) >= theta_max:
                self.C += acc
                self.__y += acc
                y_k_1 = self.__y[:-1]
                y_k = self.__y[1:]
                l_k = y_k_1 / y_k
            else:
                break

        # 生成y的等权邻值生成数列
        y1 = []
        for i in range(len(self.__y)):
            y1.append(sum(self.__y[:i+1]))
        y1 = np.array(y1, dtype=float)
        y1_k_1 = y1[:-1]
        y1_k = y1[1:]
        z1 = -0.5 * y1_k_1 - 0.5 * y1_k

        # 求解u矩阵
        z1 = np.array([z1])
        B = np.vstack((z1, np.ones_like(z1))).T
        Y = np.array([self.__y[1:]]).T
        self.u = np.linalg.lstsq(B, Y, rcond=None)[0]
        self.u = self.u.T[0]
    
    
    @classmethod
    def predict(self, t, recount=False):
        '''
        参数
        ----
        t：数或一维数组，时间
        recount：布尔类型，可选，True代表预测时间以原数据的最后一个时间点为-1重新计算，默认为Fasle
        
        返回
        ----
        数
        
        
        Parameter
        ---------
        t: num or 1D array, time
        recount: bool, callable, True means the prediction time is recalculated with the last time point of the original data as - 1, default=False
        
        Return
        ------
        num
        '''
        if type(t) == list or type(t) == tuple:
            t = np.array(t)
        elif type(t) == int or type(t) == float:
            t = np.array([t])
        
        if recount is True:
            t += self.__y.shape[0]
        
        t = t.astype(int)
        result = np.empty(t.shape)
        
        result[t==0] = self.__y[0]
        
        result[t!=0] = self.__y[0] - self.u[1] / self.u[0]
        result[t!=0] *= (np.e**(-self.u[0] * (t[t!=0])) - np.e**(-self.u[0] * (t[t!=0] - 1)))
        return result - self.C


class Leslie():
    '''
    Leslie模型
    解的形式为：Nt = M**t * N0
    其中，Nt是t时刻的个体数列表，M是Leslie矩阵
    
    参数
    ----
    N0：一维数组，各年龄层初始个体数目
    r：一维数组，各年龄层的生殖率
    s：一维数组，各年龄层到下一个年龄层的存活率
    age_range：整型，年龄段的跨度，默认为1
    
    属性
    ----
    Leslie_matrix：莱斯利矩阵
    
    
    Leslie model
    solution: Nt = M**t * N0
    Nt is the list of individuals at time 't' , M is Leslie matrix
    
    Parameters
    ----------
    N0: 1D array, initial number of individuals in each age group
    r: 1D array, reproductive rate of each age group
    s: 1D array, survival rate to next age group of each group
    age_range: int, the span of age groups, default=1
    
    Attribute
    ---------
    Leslie_matrix: Leslie matrix 
    '''
    def __init__(self, N0, r, s, age_range=1):
        self.N0 = np.array(N0)
        self.r = np.array(r)
        self.s = np.array(s)
        self.age_range = age_range
        
        Leslie_matrix = np.zeros((len(s), len(N0)))
        Leslie_matrix = np.vstack((r, Leslie_matrix))
        for i in range(1, len(N0)):
            Leslie_matrix[i][i-1] = s[i-1]
        self.Leslie_matrix = csr_matrix(Leslie_matrix)
    
    
    def predict(self, t):
        '''
        参数
        ----
        t：数或一维数组，时间
        
        返回
        ----
        数，Nt
        
        
        Parameter
        ---------
        t: num or 1D array, time
        
        Return
        ------
        num, Nt
        '''
        t_times = t // self.age_range
        dense_Leslie_matrix = self.Leslie_matrix.todense()
        Nt = self.N0 * dense_Leslie_matrix**t_times
        return np.array(Nt)[0]


class MC():
    '''
    蒙特卡洛模拟
    
    参数
    ----
    loop：整型，可选，循环次数，默认为1000
    random_type：字符串类型或函数类型，可选，可以取'random'、'randint'或自定义函数(无参数，需输出length * dim的二维ndarray)，默认'random'
    random_state：整型，可选，随机种子
    begin：整型，可选，表示整数随机序列的开始数字，仅当random_type='randint'时起作用
    end：整型，可选，表示整数随机序列的结束数字，仅当random_type='randint'时起作用
    n_jobs：整型，可选，调用的cpu数，-1表示全部调用，默认为1
    
    属性
    ----
    loop：整型，循环次数
    random_type：字符串类型，取随机的方法
    EX：数类型，select函数返回值的数学期望
    DX：数类型，select函数返回值的方差
    history：字典类型，历史EX和DX值
    
    
    Monte Carlo
    
    Parameters
    ----------
    loop: int, callable, loop count, default = 10000
    random_type: str or function, callable, 'random', 'randint' and function(no parameter, output a 2D ndarray with size length * dim) are optional, default = 'random'
    random_state: int, callable, random seed
    begin: int, callable, beginning number of the random sequence, it's used only when random_type='randint'
    end: int, callable, end number of the random sequence, it's used only when random_type='randint'
    n_jobs: int, callable, the number of cpus called, -1 means to call all cpus

    Atrributes
    ----------
    loop: int, loop count
    random_type: str, the method of generating random number
    EX: num, mathematical expectation of return value of select function
    DX: num, variance of return value of select function
    history: dict, historical EX and DX
    '''
    def __init__(self, loop=1000, random_type='random', random_state=None,
                 begin=None, end=None, n_jobs=1):
        self.loop = loop
        self.random_type = random_type
        self.history = {'EX':[], 'DX':[]}
        self.__begin = begin
        self.__end = end
        self.n_jobs = n_jobs
        np.random.seed(random_state)
    

    def sel_ind(self, condition):
        '''
        用于快速构建条件函数
        将condition函数作用于随机序列全部元素
        任意一个状态满足condition，则会输出1，否则输出0
        
        参数
        ----
        condition：函数，判断一个状态是否符合条件的函数，要求符合则输出True，不符合则输出False

        返回
        ----
        一个新函数


        for quick construction of condition function
        apply condition function to all elements of random sequence
        if any state is qualified, 1 will be reutrn, otherwise 0 will be return
        
        Parameters
        ----------
        condition: function, a function that determines if a state is qualified. If qualified, return True, otherwise reutrn False

        Return
        ------
        a new function
        '''
        def obj(x):
            for i in x:
                if condition(i):
                    return 1
            return 0
        return obj

    
    def sel_any(self, n, condition):
        '''
        用于快速构建条件函数
        将condition函数作用于随机序列全部元素
        至少任意n个状态满足condition，则会输出1，否则输出0
        
        参数
        ----
        n：int类型
        condition：函数，判断一个状态是否符合条件的函数，要求符合则输出True，不符合则输出False

        返回
        ----
        一个新函数


        for quick construction of condition function
        apply condition function to all elements of random sequence
        if at least any_n states are qualified, 1 will be reutrn, otherwise 0 will be return
        
        Parameters
        ----------
        n: int
        condition: function, a function that determines if a state is qualified. If qualified, return True, otherwise reutrn False

        Return
        ------
        a new function
        '''
        def obj(x):
            calcu_ = 0
            for i in x:
                if condition(i):
                    calcu_ += 1
                if calcu_ >= n:
                    break
            if calcu_ >= n:
                return 1
            else:
                return 0
        return obj
    
    
    def sel_con(self, n, condition):
        '''
        用于快速构建条件函数
        将condition函数作用于随机序列全部元素
        至少连续n个状态满足condition，则会输出1，否则输出0
        
        参数
        ----
        n：int类型
        condition：函数，判断一个状态是否符合条件的函数，要求符合则输出True，不符合则输出False

        返回
        ----
        一个新函数


        for quick construction of condition function
        apply condition function to all elements of random sequence
        if at least con_n consecutive states are qualified, 1 will be reutrn, otherwise 0 will be return
        
        Parameters
        ----------
        n: int
        condtition: function, a function that determines if a state is qualified. If qualified, return True, otherwise reutrn False
        
        Return
        ------
        a new function
        '''
        def obj(x):
            calcu_ = 0
            for i in x:
                if condition(i):
                    calcu_ += 1
                else:
                    calcu_ *= 0
                if calcu_ >= n:
                    break
            if calcu_ >= n:
                return 1
            else:
                return 0
        return obj
    
    
    def __one_experiment(self, length, dim, select):
        '''
        进行一次完整的实验
        '''
        if self.random_type == 'random':
            random_list_ = np.random.rand(length, dim)

        elif self.random_type == 'randint':
            random_list_ = np.random.randint(self.__begin, self.__end + 1, size=(length, dim))
        
        else:
            random_list_ = self.random_type()

        #如果随机序列满足select，就算实验成功输出1，否则算失败输出0
        if type(select).__name__ == 'function':
            return select(random_list_)
        else:
            judge_ = 1
            for li in select:
                if li(random_list_):
                    judge_ *= 1
                else:
                    judge_ *= 0
            return judge_


    def fit(self, length, dim, select):
        '''
        进行蒙特卡洛模拟
        
        参数
        ----
        length：整型，随机序列长度
        dim：整型，随机序列元素维度
        select：条件函数或函数列表，select函数应该以length * dim的二维ndarray为自变量
            select函数用于甄别某一个随机矩阵是否符合预期，符合输出1，不符合输出0
            select函数同时也可以输出其他值以满足实际需求
            select若为函数列表，则要求每个函数只能输出1或0

        
        Simulate
        
        Parameters
        ----------
        length: int, the length of random sequence
        dim: int, dimension of elements in random sequence
        select: function or function list, select function should take 2D ndarray(length * dim) as independent variable
            select function used for Identifying whether the random matrix meets the expectation, if meets, output 1, otherwise output 0
            select function can also output other values to meet the actual demand
            if select is a list, every function in the list can only output 1 or 0
        '''
        final_propability_ = []
        if self.n_jobs == 1:
            for i in range(self.loop):
                freq_ = self.__one_experiment(length, dim, select)
                final_propability_.append(freq_)
        
        else:
            try:
                if self.n_jobs == -1:
                    pool = Pool(processes=cpu_count())
                else:
                    pool = Pool(processes=self.n_jobs)
                for i in range(self.loop):
                    freq_ = pool.apply(self.__one_experiment,
                                     (length,dim, select))
                    final_propability_.append(freq_)
            
            except:
                for i in range(self.loop):
                    freq_ = self.__one_experiment(length, dim, select)
                    final_propability_.append(freq_)
        
        self.EX = np.mean(final_propability_)
        self.DX = np.var(final_propability_)
        self.history['EX'].append(self.EX)
        self.history['DX'].append(self.DX)
    
    
    def clr(self):
        '''
        清空历史数据
        
        
        clear the historical data
        '''
        self.history = {'EX':[], 'DX':[]}


class GA():
    '''
    遗传算法
    
    参数
    ----
    population：整型，种群数
    iteration: 整型，迭代次数（自然选择次数）
    random_type：字符串类型，可选，可以取'random'和'randint'，默认'random'
    select：字符串类型或函数类型，可选，选择个体的方法，可选'rw'、'st'或者自定义函数，默认'rw'
        'rw'：基于随机接受的轮盘赌选择
        'st'：随机竞争选择
        自定义函数：函数需要有两个参数，第一个参数是一个二维ndarray，第二个参数是适应度函数
    crossover：字符串类型或函数类型，可选，交叉互换的方法，可选'uniform'、'point'或自定义函数，默认'uniform'
        'uniform'：均匀交叉
        'point'：单点及多点交叉
        自定义函数：函数只能设置一个参数，以种群（二维ndarray）作为输入
    begin：整型，可选，表示整数随机序列的开始数字，仅当random_type='randint'时起作用
    end：整型，可选，表示整数随机序列的结束数字，仅当random_type='randint'时起作用
    random_state：整型，可选，随机种子
    select_rate：浮点数类型，可选，选择率（存活率），默认0.3
    mutate_rate：浮点数类型，可选，变异率，默认0.05
    crossover_rate：浮点数类型，可选，基因交叉概率，默认0.5
    repeat：布尔类型，可选，是否允许序列元素重复，默认为True，仅在random_type='randint'且crossover不是自定义函数时起作用
    
    属性
    ----
    population：整型，种群数
    iteration：整型，迭代次数（自然选择次数）
    random_type：字符串类型，取随机的方法
    select：字符串类型，选择个体的方法
    crossover：字符串类型，交叉互换的方法
    dom：二维ndarray，优势种群
    dom_fitness：一维ndarray，优势种群的适应度
    
    
    Genetic Algorithm
    
    Parameters
    ----------
    population: int, the population size
    iteration: int, the times of iterations( the times of natural selection)
    random_type: str, callable, 'random' and 'randint' are optional, default = 'random'
    select: string or function, callable, the method of selecting individuals, 'rw', 'st' or custom function, default = 'rw'
        'rw': Roulette Wheel Selection
        'st': Stochastic Tournament Selection
        custom function: the function needs two parameters, the first is a 2D ndarray, and the second is the fitness function
    crossover: string or function, callable, the method of crossing over, 'uniform', 'point' or custom function, default = 'uniform'
        'uniform': Uniform Crossover
        'point': Multi-point Crossover
        custom function: the function can only set one parameter with population (2D ndarray) as input
    begin: int, callable, beginning number of the random sequence, it's used only when random_type='randint'
    end: int, callable, end number of the random sequence, it's used only when random_type='randint'
    random_state: int, callable, random seed
    select_rate: float, callable, selection rate( survival rate), default=0.3
    mutate_rate: float, callable, variation rate, default=0.05
    crossover_rate: float, callable, gene crossover probability, default=0.5
    repeat: bool, callable, whether sequence elements are allowed to repeat, default=True, only works when random_type='randint' and crossover is not a custom function

    Atrributes
    ----------
    population: int, the population size
    iteration: int, the times of iterations( the times of natural selection)
    random_type: str, the method of generating random number
    select: str, the method of selecting individuals
    crossover: str, the method of crossing over
    dom: 2D ndarray, the dominance
    dom_fitness: 1D ndarray, the fitness of the dominance
    '''
    def __init__(self, population=1000, iteration=100, random_type='random',
                 select='rw', crossover='uniform', begin=None, end=None,
                 random_state=None, select_rate=0.3, mutate_rate=0.05,
                 crossover_rate=0.5, repeat=True):
        self.population = population
        self.iteration = iteration
        self.random_type = random_type
        self.select = select
        self.crossover = crossover
        self.__begin = begin
        self.__end = end
        self.__select_rate = select_rate
        self.__mutate_rate = mutate_rate
        self.__crossover_rate = crossover_rate
        self.__repeat = repeat
        np.random.seed(random_state)
    
    
    def __mutate_func(self, length, populations_matrix):
        '''
        完成对新一代的变异
        '''
        mutate_matrix = np.random.rand(self.population, length) - self.__mutate_rate
        mutate_matrix = np.argwhere(mutate_matrix <= 0)

        if self.random_type == 'random':
            for i in mutate_matrix:
                populations_matrix[i[0], i[1]] = random.random()
        
        elif self.random_type == 'randint':
            if self.__repeat:
                for i in mutate_matrix:
                    populations_matrix[i[0], i[1]] = random.randint(self.__begin, self.__end)
                
            else:
                for i in mutate_matrix:
                    random_value = random.randint(self.__begin, self.__end)
                    index = np.where(populations_matrix[i[0]]==random_value)[0]
                    if len(index) == 0:
                        populations_matrix[i[0], i[1]] = random_value
                    else:
                        populations_matrix[i[0], index[0]] = populations_matrix[i[0], i[1]]
                        populations_matrix[i[0], i[1]] = random_value
        
        return populations_matrix
    
    
    def __repeat_adjust(self, child_individual_1, child_individual_2, random_loc_list):
        '''
        调整序列使得序列不存在重复元素，仅在crossover不是自定义函数时有效果
        '''
        mask = np.ones(child_individual_1.shape[0], bool)
        mask[random_loc_list] = False
        d_loc_list = np.where(mask)[0]
        child_11 = child_individual_1[random_loc_list].copy()    # 被交叉互换的片段
        child_12 = child_individual_1[d_loc_list].copy()         # 未被交叉互换的片段
        child_21 = child_individual_2[random_loc_list].copy()    # 被交叉互换的片段
        child_22 = child_individual_2[d_loc_list].copy()         # 未被交叉互换的片段

        for i in range(len(child_12)):
            while child_12[i] in child_11:
                index = np.where(child_11 == child_12[i])[0][0]
                child_12[i] = child_21[index]
            child_individual_1[d_loc_list] = child_12
        
        for i in range(len(child_22)):
            while child_22[i] in child_21:
                index = np.where(child_21 == child_22[i])[0][0]
                child_22[i] = child_11[index]
            child_individual_2[d_loc_list] = child_22
        
        return child_individual_1, child_individual_2
    
    
    def __crossover(self, num_point, length, parent_matrix, func_type):
        '''
        交叉
        '''
        if num_point:
            if num_point > length:
                raise Exception("'num_point' should be less than 'length'.")
        
        num_of_parents = len(parent_matrix)
        child_matrix = []
        child_population = self.population - num_of_parents
        
        while len(child_matrix) < child_population:
            while True:
                random_num_1 = random.randint(0, num_of_parents - 1)
                random_num_2 = random.randint(0, num_of_parents - 1)
                if random_num_1 != random_num_2:
                    break
            child_individual_1 = parent_matrix[random_num_1]
            child_individual_2 = parent_matrix[random_num_2]
            
            child_individual_1, child_individual_2, random_loc_list = func_type(num_point, length, child_individual_1, child_individual_2)
            if not self.__repeat:
                child_individual_1, child_individual_2 = self.__repeat_adjust(child_individual_1, child_individual_2, random_loc_list)
            
            child_matrix.append(child_individual_1)
            child_matrix.append(child_individual_2)
        child_matrix = np.array(child_matrix)
        child_matrix = np.concatenate([parent_matrix, child_matrix])
        return child_matrix
    

    def __point_crossover(self, num_point, length, child_individual_1, child_individual_2):
        '''
        多点交叉
        
        
        Multi-point Crossover
        '''
        random_loc_list = []
        for j in range(num_point):
            while True:
                random_loc = random.randint(0, length - 1)
                if random_loc not in random_loc_list:
                    random_loc_list.append(random_loc)
                    break
            medium_gene = child_individual_1[random_loc]
            child_individual_1[random_loc] = child_individual_2[random_loc]
            child_individual_2[random_loc] = medium_gene
        return child_individual_1, child_individual_2, random_loc_list
    
    
    def __uniform_crossover(self, num_point, length, child_individual_1, child_individual_2):
        '''
        均匀交叉
        
        
        Uniform Crossover
        '''
        random_loc_list = []
        for j in range(length):
            crossover_possibility = random.random()
            if crossover_possibility <= self.__crossover_rate:
                medium_gene = child_individual_1[j]
                child_individual_1[j] = child_individual_2[j]
                child_individual_2[j] = medium_gene
                random_loc_list.append(j)
        return child_individual_1, child_individual_2, random_loc_list
    
    
    def __st(self, parent_matrix, fitness, num_dead, param):
        '''
        随机竞争选择
        num_dead: 整型，死亡的数量
        
        
        Stochastic Tournament
        num_dead: int, the number of dead individual
        '''
        num_of_parents = len(parent_matrix)
        
        for i in range(num_dead):
            while True:
                random_num_1 = random.randint(0, num_of_parents - i - 1)
                random_num_2 = random.randint(0, num_of_parents - i - 1)
                if random_num_1 != random_num_2:
                    break
            if fitness(parent_matrix[random_num_1], **param) <= fitness(parent_matrix[random_num_2], **param):
                parent_matrix = np.delete(parent_matrix, random_num_1, axis=0)
            else:
                parent_matrix = np.delete(parent_matrix, random_num_2, axis=0)
        return parent_matrix
    
    
    def __rw(self, parent_matrix, fitness, num_alive, param):
        '''
        轮盘赌选择
        
        
        Roulette Wheel Selection
        '''
        fitness_list = []
        child_matrix = []
        
        for parent_individual in parent_matrix:
            fitness_list.append(fitness(parent_individual, **param))
        max_fitness = max(fitness_list)
        
        while len(child_matrix) < num_alive:
            ind = np.random.randint(0, len(parent_matrix))
            a = np.random.rand()
            if np.random.rand() > fitness_list[ind] / max_fitness or\
                parent_matrix[ind].tolist() in np.array(child_matrix).tolist():
                pass
            else:
                child_matrix.append(parent_matrix[ind])
        parent_matrix = np.array(child_matrix)
        return parent_matrix
    
    
    def fit(self, length, fitness, param={}):
        '''
        进行遗传算法模拟
        
        参数
        ----
        length：整型，染色体长度
        fitness：函数类型，适应度函数
        param：字典类型，可选，当fitness函数有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        
        
        Simulate
        
        Parameters
        ----------
        length: int, the length of chromosome
        fitness: funtion, fitness function
        param: dict, callable, when function "fitness" has other non-default parameters, "param" needs to be input as a dictionary with parm_name as key and param_value as value, default={}
        '''
        self.__fitness = fitness
        if self.random_type == 'random':
            parent_matrix = np.random.rand(self.population, length)
        elif self.random_type == 'randint':
            if not self.__repeat:
                parent_matrix = []
                for i in range(self.population):
                    parent_matrix.append(random.sample(range(self.__begin, self.__end+1), length))
                parent_matrix = np.array(parent_matrix)
            
            else:
                parent_matrix = np.random.randint(self.__begin, self.__end, size=(self.population, length))
        else:
            raise Exception("'random_type' must be one of 'random' and 'randint'.")
        
        num_alive = self.population * self.__select_rate
        num_dead = int(self.population - num_alive)
        num_point = int(length * self.__crossover_rate)
        
        for i in range(self.iteration - 1):
            # 选择
            if self.select == 'rw':
                parent_matrix = self.__rw(parent_matrix, fitness, num_alive, param)
            elif self.select == 'st':
                parent_matrix = self.__st(parent_matrix, fitness, num_dead, param)
            elif type(self.select).__name__ == 'function':
                parent_matrix == self.select(parent_matrix, fitness)
            
            # 交叉互换
            if self.crossover == 'uniform':
                parent_matrix = self.__crossover(None, length,
                                                 parent_matrix, self.__uniform_crossover)
            elif self.crossover == 'point':
                parent_matrix = self.__crossover(num_point, length,
                                                 parent_matrix, self.__point_crossover)
            
            elif type(self.crossover).__name__ == 'function':
                parent_matrix = self.crossover(parent_matrix)
            
            # 变异
            parent_matrix = self.__mutate_func(length, parent_matrix)
        
        if self.select == 'rw':
            parent_matrix = self.__rw(parent_matrix, fitness, num_alive, param)
        elif self.select == 'st':
            parent_matrix = self.__st(parent_matrix, fitness, num_dead, param)
        elif type(self.select).__name__ == 'function':
            parent_matrix == self.select(parent_matrix, fitness)
        
        self.dom = parent_matrix
        
        dominance_fitness = []
        for individual in parent_matrix:
            dominance_fitness.append(fitness(individual))
        self.dom_fitness = np.array(dominance_fitness)


class SA():
    '''
    模拟退火算法
    默认以评估函数select的最小值为最优值
    
    参数
    ----
    anneal：浮点数类型或函数类型，可选，退火方法，若为浮点数，则按T = anneal * T退火，默认为0.9
    step：浮点数类型或函数类型，可选
        当为浮点数类型时，是步长倍率，每次生成的步长为step乘一个属于(-1, 1)的随机数，默认为1
        当为函数类型时，是自变量点的更新方法
    param：字典类型，可选，当step为函数类型且有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    n：整型，可选，等温时迭代次数，默认为10
    random_state：整型，可选，随机种子
    
    属性
    ----
    solution：ndarray，最优解
    trace：ndarray，迭代过程中的自变量点
    value：浮点数类型，最优解的函数值
    
    
    Simulated Annealing Algorithm
    By default, the minimum value of the evaluation function 'select' is the optimal value
    
    Parameters
    ----------
    anneal: float or function, callable, annealing method, if type is float, it will be annealed with T = anneal * T, default=0.9
    step: float or function, callable
        when its type is float, it means step, each generated step length = step * a random number belonging to (-1, 1), default=1
        when its type is function, it menas update method of independent variable points
    param: dict, callable, When step is function and has other non-default parameters, "param" needs to be input as a dictionary with parm_name as key and param_value as value, default={}
    n: int, callable, isothermal iterations, default=10
    random_state: int, callable, random seed
    
    Attributes
    ----------
    solution: ndarray, optimal solution
    trace: ndarray, independent variable points in the iterative process
    value: float, function value of optimal solution
    '''
    def __init__(self, anneal=0.9, step=1, param={}, n=10, random_state=None):
        self.__anneal = anneal
        self.__step = step
        self.__param = param
        self.__n = n
        np.random.seed(random_state)
    
    
    def fit(self, select, T, T0, initial, args={}, loop=1):
        '''
        参数
        ----
        select：函数，评估函数
        T：浮点数类型，初始温度
        T0：浮点数类型，退火温度
        initial：数或数组，初始解，select函数的输入值
        args：字典类型，可选，当select有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        loop：整型，可选，当需要循环m次模拟退火时，loop=m，默认loop=1
        
        
        Parameters
        ----------
        select: function, evaluation function
        T: float, initial temperature
        T0: float, annealing temperature
        initial: num or array, initial solution, the input value of select function
        args: dict, callable, when function "select" has other non-default parameters, "args" needs to be input as a dictionary with parm_name as key and param_value as value, default={}
        loop: int, callable, when m cycles of simulated annealing are needed, loo=m, default=1
        '''
        initial = np.array(initial, dtype=float)
        
        for n in range(loop):
            T_copy = T
            trace = [initial]
            while T_copy > T0:
                for i in range(self.__n):
                    if type(self.__step).__name__ != 'function':
                        random_x = 2 * self.__step * (np.random.rand(*initial.shape) - 0.5)
                        initial_copy = initial.copy()
                        initial_copy += random_x
                    else:
                        initial_copy = self.__step(initial, **self.__param)
                    
                    if select(initial_copy) < select(initial, **args):
                        initial = initial_copy
                        trace.append(initial)
                    else:
                        pro = random.random()
                        if pro < np.e**(-(select(initial_copy, **args) - select(initial, **args)) / T_copy):
                            initial = initial_copy
                            trace.append(initial)
                
                if type(self.__anneal).__name__ == 'float':
                    T_copy *= self.__anneal
                elif type(self.__anneal).__name__ == 'function':
                    T_copy = self.__anneal(T_copy)
                else:
                    raise Exception("Type of 'anneal' must be one of 'float' and 'function'.")
            
            if n == 0:
                self.solution = initial
                self.trace = np.array(trace)
                self.value = select(initial, **args)
            
            else:
                value = select(initial, **args)
                if value < self.value:
                    self.solution = initial
                    self.trace = np.array(trace)
                    self.value = value


class GD():
    '''
    梯度下降算法
    沿函数负梯度方向逐步下降进而得到函数的最优解，最优解默认为最小值
    
    参数
    ----
    ytol：浮点数类型，可选，连续两次迭代的函数值小于ytol时即停止迭代，默认为0.01
    acc：浮点数类型，可选，求导精度，默认为0.1
    step：浮点数类型，可选，步长倍率，每次生成的步长为step * 负梯度，若auto=True，则步长为x*10^step，其中x为梯度的第一位有效数字，默认为0.1
    auto：布尔类型，可选，True表示采取自适应步长，默认为False
    
    属性
    ----
    solution：浮点数类型，最优解
    trace：ndarray，迭代过程中的自变量点
    value：浮点数类型，最优解的函数值
    
    
    Gradient Descent
    The optimal solution of the function is obtained by gradually decreasing along the negative gradient direction
    The optimal solution is the minimum value by default
    
    Parameters
    ----------
    ytol: float, callable, when △f of two successive iterations is less than ytol, the iteration will stop, default=0.01
    acc: float, callable, accuracy of derivation, default=0.1
    step: float, callable, step, each generated step length = - step * gradient, default=1
    auto：bool, callable
    
    Attributes
    ----------
    solution: float, optimal solution
    trace: ndarray, independent variable points in the iterative process
    value: float, function value of optimal solution
    '''
    def __init__(self, ytol=0.01, acc=0.1, step=0.1, auto=False):
        self.ytol = ytol
        self.acc = acc
        self._step = step
        self._auto = auto
    
    
    def fit(self, select, initial, param={}):
        '''
        参数
        ----
        select：函数类型，评估函数
        initial：数或数组，初始解，select函数的输入值
        param：字典类型，可选，当select有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
        
        Parameters
        ----------
        select: function, evaluation function
        initial: num or array, initial solution, the input value of select function
        param: dict, callable, when function "select" has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
        '''
        initial = np.array(initial, dtype=float)
        self.trace = [initial]
        f_change = float('inf')
        
        while f_change > self.ytol:
            d_list = grad(select, initial, self.acc)
            if self._auto:
                self._step = -1 * (np.floor(np.log(d_list) / np.log(10)).astype(float)) + self._step
                self._step = 10**self._step
            # 计算函数值变化量
            f_change = select(initial, **param)
            initial = initial - d_list * self._step
            f_change = abs(select(initial, **param) - f_change)
            self.trace.append(initial)
        
        self.solution = initial
        self.trace = np.array(self.trace)
        self.value = select(initial, **param)