'''
本模块储存着几种常用坐标变换

This module stores several common coordinate transformations
'''
import numpy as np


def lorentz(v, x):
    '''
    洛伦兹坐标变换
    
    参数
    ----
    v：惯性系的相对速度，当速度方向沿着x轴正方向时（正变换）v为正
    x：一维或二维列表，(x, y, z, t)
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Lorentz coordinate transformation, S ←→ S'
    
    Parameters
    ----------
    v: relative velocity of inertial system, 'v' is positive when the velocity direction is along the positive direction of x-axis (positive transformation)
    x: 1-D or 2-D list, (x, y, z, t)
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=np.float)
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
    v：惯性系的相对速度，当速度方向沿着x轴正方向时（正变换）v为正
    vo：一维或二维列表，(vx, vy, vz)
    
    返回
    ----
    ndarray，转换后的速度
    
    
    Lorentz speed transformation, S ←→ S'
    
    Parameters
    ----------
    v: relative velocity of inertial system, 'v' is positive when the velocity direction is along the positive direction of x-axis (positive transformation)
    vo: 1-D or 2-D list, (vx, vy, vz)
    
    Return
    ------
    ndarray, converted velocity
    '''
    vo = np.array(vo, dtype=np.float)
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
    x = np.array(x, dtype=np.float)
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
    x=np.array(x, dtype=np.float)
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
    x = np.array(x, dtype=np.float)
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
    x = np.array(x)
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
    
    参数
    ----
    x：一维或二维列表，(x, y)
    theta：浮点数类型，坐标系绕原点逆时针旋转的角度
    
    返回
    ----
    ndarray，转换后的坐标
    
    
    Rotation transformation of plane rectangular coordinate system
    'theta' is positive when rotating anticlockwise and negative when rotating clockwise
    
    Parameter
    ---------
    x: 1-D or 2-D list, (x, y)
    theta: float, the angle that the coordinate system rotates counterclockwise about the origin
    
    Return
    ------
    ndarray, converted coordinates
    '''
    x = np.array(x, dtype=np.float)
    if len(x.shape) == 1:
        x0 = np.cos(theta) * x[0] + np.sin(theta) * x[1]
        x1 = np.cos(theta) * x[1] - np.sin(theta) * x[0]
        x[0], x[1] = x0, x1
    
    else:
        x0 = np.cos(theta) * x[:, 0] + np.sin(theta) * x[:, 1]
        x1 = np.cos(theta) * x[:, 1] - np.sin(theta) * x[:, 0]
        x[:, 0], x[:, 1] = x0, x1
    
    return x