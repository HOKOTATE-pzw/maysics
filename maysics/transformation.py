'''
本模块储存着几种常用坐标变换

This module stores several common coordinate transformations
'''

import numpy as np
from maysics import constant


def lorentz(v, x):
    '''
    洛伦兹坐标变换
    
    参数
    ----
    v：惯性系的相对速度，当速度方向沿着x轴正方向时（正变换）v为正
    x：列表，(x, y, z, t)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Lorentz coordinate transformation, S ←→ S'
    
    Parameters
    ----------
    v: relative velocity of inertial system, 'v' is positive when the velocity direction is along the positive direction of x-axis (positive transformation)
    x: list, (x, y, z, t)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
    γ = 1 / (1 - (v / constant.c)**2)**0.5
    xp = γ * (x[0] - v * x[3])
    yp = x[1]
    zp = x[2]
    tp = γ * (x[3] - v * x[0] / constant.c**2)
    
    return np.array([xp, yp, zp, tp])
    

def lorentz_v(v, vo):
    '''
    洛伦兹速度变换
    
    参数
    ----
    v：惯性系的相对速度，当速度方向沿着x轴正方向时（正变换）v为正
    vo：列表，(vx, vy, vz)
    
    返回
    ----
    1-D ndarray，转换后的速度
    
    
    Lorentz speed transformation, S ←→ S'
    
    Parameters
    ----------
    v: relative velocity of inertial system, 'v' is positive when the velocity direction is along the positive direction of x-axis (positive transformation)
    vo: list, (vx, vy, vz)
    
    Return
    ------
    1-D ndarray, converted velocity
    '''
    γ = 1 / (1 - (v / constant.c)**2)**0.5
    factor = 1 - v * vo[0] / constant.c**2
    vxp = (vo[0] - v) / factor
    vyp = vo[1] * γ / factor
    vzp = vo[2] * γ / factor
    
    return np.array([vxp, vyp, vzp])


def polar(x):
    '''
    极坐标正变换
    
    参数
    ----
    x：列表，(x, y)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Polar positive transformation
    
    Parameters
    ----------
    x: list, (x, y)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
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
    
    return np.array([r, theta])


def ipolar(xp):
    '''
    极坐标逆变换
    
    参数
    ----
    xp：列表，(r, θ)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Parameters
    ----------
    xp: list, (r, θ)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
    x = xp[0] * np.cos(xp[1])
    y = xp[0] * np.sin(xp[1])
    
    return np.array([x, y])


def cylinder(x):
    '''
    柱坐标正变换
    
    参数
    ----
    x：列表，(x, y, z)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Cylinder positive transformation
    
    Parameters
    ----------
    x: list, (x, y, z)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
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
    z = x[2]
    
    return np.array([r, theta, z])


def icylinder(xp):
    '''
    柱坐标逆变换
    
    参数
    ----
    xp：列表，(r, θ, z)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Cylinder inverse transformation
    
    Parameters
    ----------
    xp: list, (r, θ, z)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
    x = xp[0] * np.cos(xp[1])
    y = xp[0] * np.sin(xp[1])
    z = xp[2]
    
    return np.array([x, y, z])


def sphere(x):
    '''
    球坐标正变换
    
    参数
    ----
    x：列表，(x, y, z)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Sphere positive transformation
    
    Parameters
    ----------
    x: list, (x, y, z)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
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
    
    return np.array([r, theta, phai])


def isphere(xp):
    '''
    球坐标逆变换
    
    参数
    ----
    xp：列表，(r, θ, φ)
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Sphere inverse transformation
    
    Parameters
    ----------
    xp: list, (r, θ, φ)
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
    x = xp[0] * np.sin(xp[2]) * np.cos(xp[1])
    y = xp[0] * np.sin(xp[2]) * np.sin(xp[1])
    z = xp[0] * np.cos(xp[2])
    
    return np.array([x, y, z])


def rotate(theta, x):
    '''
    平面直角坐标系的旋转变换
    逆时针旋转时theta为正，顺时针旋转时theta为负
    
    参数
    ----
    x：列表，(x, y)
    theta：浮点数类型，坐标系绕原点逆时针旋转的角度
    
    返回
    ----
    1-D ndarray，转换后的坐标
    
    
    Rotation transformation of plane rectangular coordinate system
    'theta' is positive when rotating anticlockwise and negative when rotating clockwise
    
    Parameter
    ---------
    x: list, (x, y)
    theta: float, the angle that the coordinate system rotates counterclockwise about the origin
    
    Return
    ------
    1-D ndarray, converted coordinates
    '''
    xp = np.cos(theta) * x[0] + np.sin(theta) * x[1]
    yp = np.cos(theta) * x[1] - np.sin(theta) * x[0]
    
    return np.array([xp, yp])