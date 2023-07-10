'''
本模块储存着几种常用坐标变换

This module stores several common coordinate transformations
'''
import numpy as np


def dirac(x, num=None, rl=0):
    '''
    将狄拉克算符转换为向量
    
    参数
    ----
    x：一维数组类型，数据
    num：一维数组类型，代表每一个量子位的进制，默认均为二进制
    rl：rl=0或rl='r'代表右矢，rl=1或rl='1'代表左矢
    
    返回
    ----
    二维ndarray，狄拉克算符的向量形式，右矢为列向量，左矢为行向量
    
    
    Convert Dirac Operator to Vector
    
    Parameters
    ----------
    x: 1D-array, data
    num: 1D-array, represents the base of each qubit, which is binary by default
    rl: rl=0 or rl='r' means right vector, rl=1 or rl='l' means left vector
    
    Return
    ------
    2D-ndarray, the vector form of Dirac operator, the right vector is the column vector and the left vector is the row vector
    '''
    x = np.array(x)
    total = x.shape[0]
    if num is None:
        num = np.ones_like(x) * 2
    n = 1
    for i in num:
        n *= i
    m = x[-1]
    for i in range(1, total):
        m += x[- i - 1] * np.prod(num[-i:])
    if rl == 0 or rl == 'r':
        vec = np.zeros((n, 1))
        vec[m, 0] = 1
    elif rl == 1 or rl == 'l':
        vec = np.zeros((1, n))
        vec[0, m] = 1
    return vec


def lorentz(v, x):
    '''
    洛伦兹坐标变换
    
    参数
    ----
    v：数类型，惯性系的相对速度，当速度方向沿着x轴正方向时（正变换）v为正
    x：一维或二维列表，(x, y, z, t)
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Lorentz coordinate transformation, S ←→ S'
    
    Parameters
    ----------
    v: num, relative velocity of inertial system, 'v' is positive when the velocity direction is along the positive direction of x-axis (positive transformation)
    x: 1-D or 2-D list, (x, y, z, t)
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=float)
    γ = 1 / (1 - (v / 299792458.0)**2)**0.5
    if len(x.shape) == 1:
        xp = γ * (x[0] - v * x[3])
        tp = γ * (x[3] - v * x[0] / 299792458.0**2)
        x[0], x[3] = xp, tp
    
    else:
        xp = γ * (x[:, 0] - v * x[:, 3])
        tp = γ * (x[:, 3] - v * x[:, 0] / 299792458.0**2)
        x[:, 0], x[:, 3] = xp, tp
    
    return x
    

def lorentz_v(v, vo):
    '''
    洛伦兹速度变换
    
    参数
    ----
    v：数类型，惯性系的相对速度，当速度方向沿着x轴正方向时（正变换）v为正
    vo：一维或二维列表，(vx, vy, vz)
    
    返回
    ----
    ndarray，转换后的速度
    
    
    Lorentz speed transformation, S ←→ S'
    
    Parameters
    ----------
    v: num, relative velocity of inertial system, 'v' is positive when the velocity direction is along the positive direction of x-axis (positive transformation)
    vo: 1-D or 2-D list, (vx, vy, vz)
    
    Return
    ------
    ndarray, converted velocity
    '''
    vo = np.array(vo, dtype=float)
    γ = 1 / (1 - (v / 299792458.0)**2)**0.5
    if len(vo.shape) == 1:
        factor = 1 - v * vo[0] / 299792458.0**2
        vo[0] = (vo[0] - v) / factor
        vo[1] = vo[1] * γ / factor
        vo[2] = vo[2] * γ / factor
    
    else:
        factor = 1 - v * vo[:, 0] / 299792458.0**2
        vo[:, 0] = (vo[:, 0] - v) / factor
        vo[:, 1] = vo[:, 1] * γ / factor
        vo[:, 2] = vo[:, 2] * γ / factor
    
    return vo


def mercator(lon_lat, r=6371393, re_lon=0):
    '''
    墨卡托变换
    𝑥 = 𝑅(𝜃−𝜃0)
    𝑦 = 𝑅𝑙𝑛(𝑡𝑎𝑛(0.25𝜋+0.5𝜙))
    
    参数
    ----
    lon_lat：一维或二维数组，经度、纬度
    r：数类型，可选，球体半径，默认为地球平均半径
    re_lon：数类型，可选，参考经度，默认为0
    
    返回
    ----
    ndarray类型，变换后的数组
    
    
    Mercator transformation
    𝑥 = 𝑅(𝜃−𝜃0)
    𝑦 = 𝑅𝑙𝑛(𝑡𝑎𝑛(0.25𝜋+0.5𝜙))
    
    Parameters
    ----------
    lon_lat: 1D or 2D array, longtitude and latitude
    r: num, callable, radius of the sphere, default=the mean radius of the earth
    re_lon: num, callable, reference longtitude, default=0
    
    Return
    ------
    ndarray, array after transformation
    '''
    lon_lat = np.array(lon_lat) * np.pi / 180
    re_lon * np.pi / 180
    if len(lon_lat.shape) == 1:
        result = np.array([r * (lon_lat[0] - re_lon), r * np.log(np.tan(0.25 * np.pi + 0.5 * lon_lat[1]))])
    elif len(lon_lat.shape) == 2:
        result = np.array([r * (lon_lat[:, 0] - re_lon), r * np.log(np.tan(0.25 * np.pi + 0.5 * lon_lat[:, 1]))])
    else:
        raise Exception("Parameter 'lon_lat' must be 1-D or 2-D.")
    
    return result


def imercator(x_y, r=6371393, re_lon=0):
    '''
    墨卡托逆变换
    𝜃 = 𝑥/𝑅 + 𝜃0
    𝜙 = 2𝑎𝑟𝑐𝑡𝑎𝑛(𝑒^(𝑦/𝑅)) − 0.5𝜋
    
    参数
    ----
    x_y: 一维或二维数组，坐标
    r：数类型，可选，球体半径，默认为地球平均半径
    re_lon：数类型，可选，参考经度，默认为0
    
    
    返回
    ----
    ndarray类型，变换后的数组
    
    
    Mercator inverse transformation
    𝜃 = 𝑥/𝑅+𝜃0
    𝜙 = 2𝑎𝑟𝑐𝑡𝑎𝑛(𝑒^(𝑦/𝑅))−0.5𝜋
    
    Parameters
    ----------
    x_y: 1D or 2D array, location
    r: num, callable, radius of the sphere, default=the mean radius of the earth
    re_lon: num, callable, reference longtitude, default=0
    
    Return
    ------
    ndarray, array after transformation
    '''
    x_y = np.array(x_y)
    if len(x_y.shape) == 1:
        result = [(x_y[0] / r) * 180 / np.pi + re_lon, (np.arctan(np.e**(x_y[1] / r)) - 0.25 * np.pi) * 360 / np.pi]
    elif len(x_y.shape) == 2:
        result = [(x_y[:, 0] / r) * 180 / np.pi + re_lon, (np.arctan(np.e**(x_y[:, 1] / r)) - 0.25 * np.pi) * 360 / np.pi]
    else:
        raise Exception("Parameter 'x_y' must be 1-D or 2-D.")
    
    return np.array(result)


def polar(x):
    '''
    极坐标或柱坐标正变换
    
    参数
    ----
    x：一维或二维列表，(x, y)或(x, y, z)
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Polar or Cylinder positive transformation
    
    Parameters
    ----------
    x: 1-D or 2-D list, (x, y) or (x, y, z)
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=float)
    if len(x.shape) == 1:
        r = (x[0]**2 + x[1]**2)**0.5
        if x[0] == 0:
            if x[1] > 0:
                theta = np.pi/2
            elif x[1] < 0:
                theta = -np.pi/2
            elif x[1] == 0:
                theta = 0
        else:
            theta = np.arctan(x[1] / x[0])
            if x[0] < 0 and x[1] > 0:
                theta += np.pi
            elif x[0] < 0 and x[1] < 0:
                theta -= np.pi
        x[0], x[1] = r, theta
    
    else:
        r = (x[:, 0]**2 + x[:, 1]**2)**0.5
        index1 = np.where(x[:, 0] != 0)[0]
        index2 = np.where(x[:, 0] == 0)[0]
        index3 = np.all([x[:, 0] < 0, x[:, 1] > 0], axis=0)
        index3 = np.where(index3 == True)[0]
        index4 = np.all([x[:, 0] < 0, x[:, 1] < 0], axis=0)
        index4 = np.where(index4 == True)[0]
        
        x[index1, 1] = np.arctan(x[index1, 1] / x[index1, 0])
        x[index3, 1] += np.pi
        x[index4, 1] -= np.pi
        
        x_new = x[index2, 1]
        x_new[np.where(x_new > 0)[0]] = np.pi/2
        x_new[np.where(x_new < 0)[0]] = -np.pi/2
        x[index2, 1] = x_new
        x[:, 0] = r
    
    return x


def ipolar(x):
    '''
    极坐标或柱坐标逆变换
    
    参数
    ----
    x：一维或二维列表，(r, θ)或(r, θ, z)
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Polar or Cylinder inverse transformation
    
    Parameters
    ----------
    x: 1-D or 2-D list, (r, θ) or (r, θ, z)
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x=np.array(x, dtype=float)
    if len(x.shape) == 1:
        x0 = x[0] * np.cos(x[1])
        x1 = x[0] * np.sin(x[1])
        x[0], x[1] = x0, x1
    
    else:
        x0 = x[:, 0] * np.cos(x[:, 1])
        x1 = x[:, 0] * np.sin(x[:, 1])
        x[:, 0], x[:, 1] = x0, x1
    
    return x


def sphere(x):
    '''
    球坐标正变换
    
    参数
    ----
    x：一维或二维列表，(x, y, z)
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Sphere positive transformation
    
    Parameters
    ----------
    x: 1-D or 2-D list, (x, y, z)
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=float)
    if len(x.shape) == 1:
        r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
        if r == 0:
            theta = 0
            phai = 0
        else:
            phai = np.arccos(x[2] / r)
            if x[0] == 0:
                if x[1] > 0:
                    theta = np.pi/2
                elif x[1] < 0:
                    theta = -np.pi/2
                elif x[1] == 0:
                    theta = 0
            else:
                theta = np.arctan(x[1] / x[0])
        x[0], x[1], x[2] = r, theta, phai
    
    else:
        r = (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5
        index_r1 = np.where(r == 0)[0]
        x[index_r1] = 0
        
        index_r2 = np.where(r != 0)[0]
        x_new = x[index_r2]
        r_new = r[index_r2]
        x_new[:, 2] = np.arccos(x_new[:, 2] / r_new)    # 处理了phi
        
        index1 = np.where(x_new[:, 0] != 0)[0]
        index2 = np.where(x_new[:, 0] == 0)[0]
        
        x_new[index1, 1] = np.arctan(x_new[index1, 1] / x_new[index1, 0])
        x_new_new = x_new[index2, 1]
        x_new_new[np.where(x_new_new > 0)] = np.pi/2
        x_new_new[np.where(x_new_new < 0)] = -np.pi/2
        x_new[index2, 1] = x_new_new
        x_new[:, 0] = r_new
        x[index_r2] = x_new             # 处理了theta和r
    
    return x


def isphere(x):
    '''
    球坐标逆变换
    
    参数
    ----
    x：一维或二维列表，(r, θ, φ)
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Sphere inverse transformation
    
    Parameters
    ----------
    x: 1-D or 2-D list, (r, θ, φ)
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=float)
    if len(x.shape) == 1:
        x0 = x[0] * np.sin(x[2]) * np.cos(x[1])
        x1 = x[0] * np.sin(x[2]) * np.sin(x[1])
        x2 = x[0] * np.cos(x[2])
        x[0], x[1], x[2] = x0, x1, x2
    
    else:
        x0 = x[:, 0] * np.sin(x[:, 2]) * np.cos(x[:, 1])
        x1 = x[:, 0] * np.sin(x[:, 2]) * np.sin(x[:, 1])
        x2 = x[:, 0] * np.cos(x[:, 2])
        x[:, 0], x[:, 1], x[:, 2] = x0, x1, x2
    
    return x


def rotate(theta, x):
    '''
    平面直角坐标系的旋转变换
    逆时针旋转时theta为正，顺时针旋转时theta为负
    𝑥 = 𝑐𝑜𝑠(𝜃)𝑥 + 𝑠𝑖𝑛(𝜃)𝑦
    𝑦 = 𝑐𝑜𝑠(𝜃)𝑥 − 𝑠𝑖𝑛(𝜃)𝑦
    
    参数
    ----
    x：一维或二维列表，(x, y)
    theta：浮点数类型，坐标系绕原点逆时针旋转的角度
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Rotation transformation of plane rectangular coordinate system
    'theta' is positive when rotating anticlockwise and negative when rotating clockwise
    𝑥 = 𝑐𝑜𝑠(𝜃)𝑥 + 𝑠𝑖𝑛(𝜃)𝑦
    𝑦 = 𝑐𝑜𝑠(𝜃)𝑥 − 𝑠𝑖𝑛(𝜃)𝑦
    
    Parameter
    ---------
    x: 1-D or 2-D list, (x, y)
    theta: float, the angle that the coordinate system rotates counterclockwise about the origin
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=float)
    if len(x.shape) == 1:
        x0 = np.cos(theta) * x[0] + np.sin(theta) * x[1]
        x1 = np.cos(theta) * x[1] - np.sin(theta) * x[0]
        x[0], x[1] = x0, x1
    
    else:
        x0 = np.cos(theta) * x[:, 0] + np.sin(theta) * x[:, 1]
        x1 = np.cos(theta) * x[:, 1] - np.sin(theta) * x[:, 0]
        x[:, 0], x[:, 1] = x0, x1
    
    return x