'''
本模块储存着一些常用的常数

This module contents some usual constants

数学 / math

chaos_1: 第一费根鲍姆常数 / the first Feigenbaum constant
chaos_2: 第二费根鲍姆常数 / the second Feigenbaum constant
e: 自然常数 / natural constant
gamma: 欧拉-马歇罗尼常数 / Euler-Mascheroni constant
golden: 黄金比例 / golden ratio
K: 兰道-拉马努金常数 / Landau–Ramanujan constant
K0: 辛钦常数 / Khintchine constant
pi: 圆周率 / Ratio of circumference to diameter
lp(): 勒让德多项式 / Legendre Polynomials
lpN()：勒让德多项式的模 / Module of Legendre Polynomials


物理 / physics

AU: 天文单位 / astronomical unit
atm: 标准大气压 / standard atmospheric pressure
c: 真空光速 / the speed of light in vacuum
c_e: 元电荷 / elementary charge
epsilon: 真空介电常数 / permittivity of vacuum
g: 重力加速度 / gravitational acceleration
G: 万有引力常数 / gravitational constant
h: 普朗克常数 / Planck constant
hr: 约化普朗克常数 / reduced Planck constant
k: 玻尔兹曼常数 / Boltzmann constant
lambdac: 康普顿波长 / Compton wavelength
ly: 光年 / light-year
m_e: 电子质量 / mass of electron
m_earth: 地球质量 / mass of the earth
m_n: 中子质量 / mass of neutron
m_p: 质子质量 / mass of proton
m_s: 太阳质量 / mass of the sun
miu: 真空磁导率 / permeability of vacuum
NA: 阿伏伽德罗常数 / Avogadro constant
pc: 秒差距 / Parsec
Platonic_year: 柏拉图年 / Platonic year
R: 理想气体常数 / gas constant
r_earth: 地球平均半径 / mean radius of the earth
r_sun: 太阳平均半径 / mean radius of the sun
r_e_m: 地月平均距离 / mean distance between the earth and the moon
SB: 斯特藩-玻尔兹曼常量 / Stefan-Boltzmann constant
v1: 第一宇宙速度 / first cosmic velocity
v2: 第二宇宙速度 / second cosmic velocity
v3: 第三宇宙速度 / third cosmic velocity
hp()：厄米多项式 / Hermite Polynomials
'''
import numpy as np
from scipy.special import factorial


# math
pi = 3.141592653589793
e = 2.718281828459045
golden = 1.618033988749895
gamma = 0.57721566490153286060651209
K = 0.76422365358922066299069873125
chaos_1 = 4.669201609102990
chaos_2 = 2.502907875095892
K0 = 2.6854520010


def lpN(l):
    '''
    l阶勒让德多项式的模
    
    参数
    ----
    l：整型，阶数
    
    返回
    ----
    数类型
    
    
    Module of Legendre Polynomials at degree l
    
    Paramater
    ---------
    l: int, degree
    
    Return
    ------
    num
    '''
    return (2 / (2 * l + 1))**0.5
    

def lp(x, l):
    '''
    l阶勒让德多项式的值
    
    参数
    ----
    x：数，输入值
    l：整型，阶数
    
    返回
    ----
    数
    
    
    Value of Legendre Polynomials at degree l
    
    Parameters
    ----------
    x: num, input value
    l: int, degree
    
    Return
    ------
    num
    '''
    result = 0
    for k in range(int(l/2) + 1):
        result += (-1)**k * factorial(2 * l - 2 * k)\
        / 2**l / factorial(k) / factorial(l - k) / factorial(l - 2 * k)\
        * x**(l - 2 * k)
    return result


# physics
atm = 101325
c = 299792458.0
c_e = 1.602176634e-19
m_e = 9.1093837015e-31
m_earth = 5.965e24
m_n = 1.67492749804e-27
m_p = 1.67262192369e-27
m_sun = 1.9891e30
G = 6.6743e-11
g = 9.80665
h = 6.62607015e-34
hr = 1.0545718176461565e-34
k = 1.380649e-23
miu = 1.2566370614359173e-06
epsilon = 8.85e-12
NA = 6.02214179e23
R = 8.31446114959365
r_earth = 6371393
r_sun = 6.955e8
r_moon = 384403000
lambdac = 2.4263102175e-12
ly = 9460730472580800
AU = 149597870700
pc = 3.0856775814671915808e16
Platonic_year = 26000
SB = 5.670367e-8
v1 = 7900
v2 = 11200
v3 = 16700


def hp(x, n, li=False):
    '''
    n阶厄米多项式

    参数
    ----
    x：数，输入值
    n：整型，阶数
    li：布尔类型，可选，True表示将0到n阶勒让德多项式组成一维ndarray返回，默认为False

    返回
    ----
    数或一维ndarray


    Value of Hermite Polynomials at degree n

    Parameters
    ----------
    x: num, input value
    n: int, degree
    li: bool, callable, True means that 0 to n order Legendre polynomials are formed into 1-D ndarray and returned, default=False
    
    Reuturn
    -------
    num or 1-D ndarray
    '''
    if n <= 0:
        n = 0
    
    if not li:
        if n == 0:
            return 1

        elif n == 1:
            return 2 * x

        else:
            return 2 * x * hp(x, n-1) - 2 * (n-1) * hp(x, n-2)
    
    else:
        hp_list = []
        for i in range(n+1):
            if i == 0:
                hp_list.append(1)
            
            elif i == 1:
                hp_list.append(2 * x)
            
            else:
                hp_list.append(2 * x * hp_list[i-1] - 2 * (i-1) * hp_list[i-2])
        return np.array(hp_list)