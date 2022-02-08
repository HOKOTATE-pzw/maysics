'''
本模块储存着部分常用的模型

This module stores some commonly used models
'''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


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
    x: 1-D or 2-D ndarray, independent variable
    y: 1-D or 2-D ndarray, dependent variable
    
    Return
    ------
    1-D ndarray: coefficient row matrix
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
        t: num or 1-D array, time for prediction
        
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
        t_span：元组形式，求解的上下限
        method：字符串形式，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组形式，可选，每当t等于该数组中的值时，会生成一个数值解

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
        alpha：衡量康复者获得免疫的时间
        t_span：元组形式，求解的上下限
        method：字符串形式，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组形式，可选，每当t等于该数组中的值时，会生成一个数值解

        返回
        ----
        solve_ivp数值解，顺序是S, I, R
        
        
        SIRS:
        
        Parameters
        ----------
        gamma: num, recovery rate
        alpha: measuring the time that the recovered people with immunity
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
        t_span：元组形式，求解的上下限
        method：字符串形式，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组形式，可选，每当t等于该数组中的值时，会生成一个数值解

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
    C：数，调整级比范围时y数组的平移量（y + C）
    u：二维ndarray，列矩阵[a b].T
    predict：函数，预测函数，仅支持输入数
    
    
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
    predict: function, prediction function, only number is supported
    '''
    @classmethod
    def fit(self, y, acc=1):
        '''
        进行GM(1, 1)拟合
        
        参数
        ----
        y：一维数组
        acc：数，可选，调整级比的精度，默认为1
        
        
        Fit with GM(1, 1)
        
        Parameter
        ---------
        y: 1-D array
        acc: num, callable, accuracy of adjusting stage ratio, default=1
        '''
        # 调整级比范围
        y = np.array(y, dtype=np.float)
        n = len(y)
        y_k_1 = y[:-1]
        y_k = y[1:]
        l_k = y_k_1 / y_k
        theta_min = np.e**(-2 / (n + 1))
        theta_max = np.e**(2 / (n + 2))

        self.C = 0
        while True:
            if np.min(l_k) <= theta_min or np.max(l_k) >= theta_max:
                self.C += acc
                y += acc
                y_k_1 = y[:-1]
                y_k = y[1:]
                l_k = y_k_1 / y_k
            else:
                break
        
        # 生成y的等权邻值生成数列
        y1 = []
        for i in range(len(y)):
            y1.append(sum(y[:i+1]))
        y1 = np.array(y1, dtype=np.float)
        y1_k_1 = y1[:-1]
        y1_k = y1[1:]
        z1 = -0.5 * y1_k_1 - 0.5 * y1_k
        
        # 求解u矩阵
        z1 = np.array([z1])
        B = np.vstack((z1, np.ones_like(z1))).T
        Y = np.array([y[1:]]).T
        self.u = np.linalg.lstsq(B, Y, rcond=None)[0]
        self.u = self.u.T[0]
        
        def predict(t):
            t = int(t)
            if t == 1:
                return self.u[1] / self.u[0] - self.C
            else:
                result = y[0] - self.u[1] / self.u[0]
                result *= (np.e**(-self.u[0] * (t - 1)) - np.e**(-self.u[0] * (t - 2)))
                return result - self.C
        
        self.predict = predict


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
    N0: 1-D array, initial number of individuals in each age group
    r: 1-D array, reproductive rate of each age group
    s: 1-D array, survival rate to next age group of each group
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
        t: num or 1-D array, time
        
        Return
        ------
        num, Nt
        '''
        t_times = t // self.age_range
        dense_Leslie_matrix = self.Leslie_matrix.todense()
        Nt = self.N0 * dense_Leslie_matrix**t_times
        return np.array(Nt)[0]