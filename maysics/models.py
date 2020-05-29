'''
本模块储存着部分常用的数理物理方程、定律、数学模型等

This module stores some commonly used mathematical physical equations, laws, mathematical models, etc
'''

import numpy as np
from maysics import operator
from maysics import constant
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


class Fouriers_law(operator.Del):
    '''
    傅里叶定律/热传导定律
    热流量：J=-k▽T
    
    参数
    ----
    k：热导率
    acc：浮点数类型，可选，求导精度，默认为0.1
    
    
    Fourier's Law: J=-k▽T
    
    Parameters
    ----------
    k: coefficient of thermal conductivity
    acc: float, callable, accuracy of derivation, default=0.1
    '''
    def __init__(self, k, acc=0.1):
        self.acc = acc
        self.k = k
    
    
    def fit(self, T):
        '''
        参数
        ----
        T：温度分布函数
        
        
        Parameter
        ---------
        T: temperature distribution function
        '''
        j = operator.Del.grad(self, T)
        def j2(x):
            j1 = -self.k * j(x)
            return j1
        
        self.J = j2



class MVD_law():
    '''
    麦克斯韦速率分布律
    
    
    Maxwell's Velocity Distribution Law
    '''
    @classmethod
    def fit(self, m, T):
        '''
        参数
        ----
        m：气体分子质量, 单位：kg
        T：气体温度, 单位：K
        
        属性
        ----
        fv：速率分布函数
        v_mean：平均速率
        v_p：最概然速率
        v_rms：均方根速率
        
        
        Parameters
        ----------
        m: mass of gas molecule, unit: kg
        T: temperature of gas, unit: K
        
        Attributes
        ----------
        fv: velocity distribution function
        v_mean: average velocity
        v_p: most probable velocity
        v_rms: root-mean-square velocity
        '''
        def func_of_v(v):
            f_v_1 = 4 * np.pi * v**2
            f_v_2 = m / (2 * np.pi * constant.k * T)**1.5
            f_v_3 = np.e**(-m * v**2 / (2 * constant.k * T))
            return f_v_1 * f_v_2 * f_v_3
        
        self.fv = func_of_v
        v_part = (constant.k * T / (np.pi * m))**0.5
        self.v_mean = 8**0.5 * v_part
        self.v_p = 2**0.5 * v_part
        self.v_rms = 3**0.5 * v_part



class Plancks_law():
    '''
    普朗克黑体辐射定律
    
    属性
    ----
    Mf：频率形式的普朗克公式，频率单位：10^10 kHz
    Ml：波长形式的普朗克公式，波长单位：100nm
    
    
    Planck's Blackbody Radiation Law
    
    Attributes
    ----------
    Mf: Planck's formula in the form of frequency, unit of frequency: 10^10 kHz
    Ml: Planck's formula in the form of wave length, unit of wave length: 100 nm
    '''
    @classmethod
    def fit(self, T):
        '''
        参数
        ----
        T：黑体温度，单位：K
        
        
        Parameter
        ---------
        T: temperature of blackbody, unit: K
        '''
        h_k = constant.h / constant.k
        def Mf(f):
            f = f * 1e13
            f_1 = 2 * np.pi * constant.h * f**3 / constant.c**2
            f_2 = 1 / (np.e**(h_k * f / T) - 1)
            return f_1 * f_2
        
        def Ml(l):
            l = l * 1e-7
            l_1 = 2 * np.pi * constant.h * constant.c**2 / l**5
            l_2 = 1 / (np.e**(h_k * constant.c / (l * T)) - 1)
            return l_1 * l_2
        
        self.Mf = Mf
        self.Ml = Ml



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
    I0：初始感染者
    K：样本总数
    beta：感染率
    
    
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
    '''
    def __init__(self, I0, K, beta, R=0, E=0):
        self.I0 = I0
        self.K = K
        self.beta = beta
        self.R = R
        self.E = E
        self.S = K - I0 - R - E
    
    
    def SI(self):
        '''
        SI模型：
        该模型不需要再额外输入参数，且可得到解析解
        解的形式为：I = K*(1-(K-I0)/(K+I0+I0*np.e**(beta*K*t)))

        返回值：元组形式，(I(t), S(t))
        
        
        SI:
        no more parameters
        solution: I = K*(1-(K-I0)/(K+I0+I0*np.e**(beta*K*t)))
        
        return: tuple, (I(t), S(t))
        '''
        def obj(t):
            I_pre_down_ = self.K + self.I0 * (np.e**(self.beta * self.K * t) - 1)
            I_pre_up_ = self.K - self.I0
            I_func_ = self.K * (1 - I_pre_up_ / I_pre_down_)
            S_func_ = self.K - I_func_
            return I_func_, S_func_
        return obj
    
    
    def SIR(self, gamma, t_span, method='RK45' ,t_eval=None):
        '''
        SIR模型：
        该模型不能得到解析解，给出的是基于solve_ivp得到的数值解数组
        
        参数
        ----
        gamma：康复率
        t_span：元组形式，求解的上下限
        method：字符串形式，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组形式，可选，每当t等于该数组中的值时，会生成一个数值解

        返回值：solve_ivp数值解，顺序是I, S, R
        
        
        SIR:
        
        Parameters
        ----------
        gamma: recovery rate
        t_span: tuple, limits of solution
        method: str, callable, solving method, 'RK45', 'RK23', 'DOP835', 'Radau', 'BDF', 'LSODA' are optional
        t_eval: list, callable, when 't' equals the value in the list, it will generate a numerical solution
        
        return: numerical solution of solve_ivp, the order is I, S, R
        '''
        y0 = [self.I0, self.S, self.R]
        def epis_equas(t, x):
            y1 = self.beta * x[0] * x[1] - gamma * x[0]
            y2 = -self.beta * x[0] * x[1]
            y3 = gamma * x[0]
            return y1, y2, y3
        result = solve_ivp(epis_equas, t_span=t_span, y0=y0, method=method, t_eval=t_eval)
        return result
    
    
    def SIRS(self, gamma, alpha, t_span, method='RK45' ,t_eval=None):
        '''
        SIRS模型：
        该模型不能得到解析解，给出的是基于solve_ivp得到的数值解数组
        
        参数
        ----
        gamma：康复率
        alpha：衡量康复者获得免疫的时间
        t_span：元组形式，求解的上下限
        method：字符串形式，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组形式，可选，每当t等于该数组中的值时，会生成一个数值解

        返回值：solve_ivp数值解，顺序是I, S, R
        
        
        SIRS:
        
        Parameters
        ----------
        gamma: recovery rate
        alpha: measuring the time that the recovered people with immunity
        t_span: tuple, limits of solution
        method: str, callable, solving method, 'RK45', 'RK23', 'DOP835', 'Radau', 'BDF', 'LSODA' are optional
        t_eval: list, callable, when 't' equals the value in the list, it will generate a numerical solution
        
        return: numerical solution of solve_ivp, the order is I, S, R
        '''
        y0= [self.I0, self.S, self.R]
        def epis_equas(t, x):
            y1 = self.beta * x[0] * x[1] - gamma * x[0]
            y2 = -self.beta * x[0] * x[1] + alpha * x[2]
            y3 = gamma * x[0] - alpha * x[2]
            return y1, y2, y3
        result = solve_ivp(epis_equas, t_span=t_span, y0=y0, method=method, t_eval=t_eval)
        return result
    
    
    def SEIR(self, gamma1, gamma2, alpha, t_span, method='RK45' ,t_eval=None):
        '''
        SEIR模型：
        该模型不能得到解析解，给出的是基于solve_ivp得到的数值解数组
        
        参数
        ----
        gamma1：潜伏期康复率
        gamma2：患者康复率
        alpha：衡量康复者获得免疫的时间
        t_span：元组形式，求解的上下限
        method：字符串形式，可选，求解方法，可选择'RK45'、'RK23'、'DOP835'、'Radau'、'BDF'、'LSODA'
        t_eval：数组形式，可选，每当t等于该数组中的值时，会生成一个数值解

        返回值：solve_ivp数值解，顺序是I, S, R, E
        
        
        SEIR:
        
        Parameters
        ----------
        gamma1: recovery rate of incubation
        gamma2: recovery rate of patients
        alpha: measuring the time that the recovered people with immunity
        t_span: tuple, limits of solution
        method: str, callable, solving method, 'RK45', 'RK23', 'DOP835', 'Radau', 'BDF', 'LSODA' are optional
        t_eval: list, callable, when 't' equals the value in the list, it will generate a numerical solution
        
        return: numerical solution of solve_ivp, the order is I, S, R, E
        '''
        y0 = [self.I0, self.S, self.R, self.E]
        def epis_equas(t, x):
            y1 = alpha * x[3] - gamma2 * x[0]
            y2 = -self.beta * x[0] *x[1]
            y3 = gamma1 * x[3] + gamma2 * x[0]
            y4 = self.beta * x[0] * x[1] - (alpha + gamma1) * x[3]
            return y1, y2, y3, y4
        result = solve_ivp(epis_equas, t_span=t_span, y0=y0, method=method, t_eval=t_eval)
        return result



class Leslie():
    '''
    Leslie模型
    解的形式为：Nt = M**t * N0
    其中，Nt是t时刻的个体数列表，M是Leslie矩阵
    
    参数
    ----
    N0：列表，各年龄层初始个体数目
    r：数，各年龄层的生殖率
    s：数，各年龄层到下一个年龄层的存活率
    age_range：整型，年龄段的跨度，默认为1
    
    
    Leslie model
    solution: Nt = M**t * N0
    Nt is the list of individuals at time 't' , M is Leslie matrix
    
    Parameters
    ----------
    N0: list, initial number of individuals in each age group
    r: number, reproductive rate of each age group
    s: number, survival rate to next age group of each group
    age_range: int, the span of age groups, default=1
    '''
    def __init__(self, N0, r, s, age_range=1):
        self.N0 = np.mat(N0).T
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
        t：时间
        
        返回值：Nt
        
        
        Parameter
        ---------
        t: time
        
        return: Nt
        '''
        t_times = t // self.age_range
        dense_Leslie_matrix = self.Leslie_matrix.todense()
        Nt = dense_Leslie_matrix**t_times * self.N0
        return Nt



class Logistic():
    '''
    Logistic人口增长模型
    该模型可得到解析解
    解的形式为：Nt = K*N0/(N0+(K-N0)*np.e**(-r*t))
    其中，Nt是t时刻的人口数
    
    参数
    ----
    N0：数，现有人口数
    r：数，人口自然增长率
    K：数，环境资源允许的稳定人口数
    
    
    Logisyic population growth models
    solution: Nt = K*N0/(N0+(K-N0)*np.e**(-r*t))
    Nt is the population at time 't'
    
    Parameters
    ----------
    N0: number or list, initial population
    r: number, natural population growth rate
    K: number, stable population allowed by environmental resources
    '''
    def __init__(self, N0, r, K):
        self.N0 = N0
        self.r = r
        self.K = K
    
    
    def predict(self, t):
        '''
        参数
        ----
        t：时间
        
        返回值：Nt
        
        
        Parameter
        ---------
        t: time
        
        return: Nt
        '''
        Nt_pre_down_ = self.N0 + (self.K - self.N0) * np.e**(-self.r * t)
        Nt = self.K * self.N0 / Nt_pre_down_
        return Nt