'''
本模块储存着几种常用坐标变换

This module stores several common coordinate transformations
'''

import numpy as np
from maysics import constant


class Lorentz():
    '''
    洛伦兹变换
    
    参数
    ----
    v：速度
    
    
    Lorentz transformation
    
    Parameters
    ----------
    v: speed
    '''
    def __init__ (self, v):
        self.v = v
        self.γ = 1 / (1 - (v / constant.c)**2)**0.5
    

    def lor(self, x):
        '''
        洛伦兹坐标正变换
        
        参数
        ----
        x：列表，(x, y, z, t)
        
        coordinate transformation, S → S'
        
        Parameters
        ----------
        x: list, (x, y, z, t)
        '''
        xp = self.γ * (x[0] - self.v * x[3])
        yp = x[1]
        zp = x[2]
        tp = self.γ * (x[3] - self.v * x[0] / constant.c**2)
        
        return np.array([xp, yp, zp, tp])
    

    def ilor(self, xp):
        '''
        洛伦兹坐标逆变换
        
        参数
        ----
        xp：列表，(x', y', z', t')
        
        coordinate transformation, S' → S:
        
        Parameters
        ----------
        xp: list, (x', y', z', t')
        '''
        x = self.γ * (xp[0] + self.v * xp[3])
        y = xp[1]
        z = xp[2]
        t = self.γ * (xp[3] + self.v * xp[0] / constant.c**2)
        
        return np.array([x, y, z, t])
    

    def vel(self, vo):
        '''
        洛伦兹速度正变换
        
        参数
        ----
        vo：列表，(vx, vy, vz)
        
        speed transformation, S → S'
        
        Parameters
        ----------
        v0: list, (vx, vy, vz)
        '''
        factor = 1 - self.v * vo[0] / constant.c**2
        vxp = (vo[0] - self.v) / factor
        vyp = vo[1] * self.γ / factor
        vzp = vo[2] * self.γ / factor
        
        return np.array([vxp, vyp, vzp])
    

    def ivel(self, vp):
        '''
        洛伦兹速度逆变换
        
        参数
        ----
        vp：列表，(vx', vy', vz')
        
        speed transformation, S' → S
        
        Parameters
        ----------
        vp: list, (vx', vy', vz')
        '''
        factor = 1 + self.v * vp[0] / constant.c**2
        vxo = (vp[0] + self.v) / factor
        vyo = vp[1] * self.γ / factor
        vzo = vp[2] * self.γ / factor
        
        return np.array([vxo, vyo, vzo])


class Polar():
    '''
    极坐标系与直角坐标系转换
    
    
    Polar transformation
    '''
    @classmethod
    def pol(self, x):
        '''
        极坐标正变换
        
        参数
        ----
        x：列表，(x, y)
        
        
        Parameters
        ----------
        x: list, (x, y)
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


    @classmethod
    def ipol(self, xp):
        '''
        极坐标逆变换
        
        参数
        ----
        xp：列表，(r, θ)
        
        
        Parameters
        ----------
        xp: list, (r, θ)
        '''
        x = xp[0] * np.cos(xp[1])
        y = xp[0] * np.sin(xp[1])
        
        return np.array([x, y])


class Cylinder():
    '''
    柱坐标系与直角坐标系转换
    
    
    Cylinder transformation
    '''
    @classmethod
    def cyl(self, x):
        '''
        柱坐标正变换
        
        参数
        ----
        x：列表，(x, y, z)
        
        
        Parameters
        ----------
        x: list, (x, y, z)
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


    @classmethod
    def icyl(self, xp):
        '''
        柱坐标逆变换
        
        参数
        ----
        xp：列表，(r, θ, z)
        
        
        Parameters
        ----------
        xp: list, (r, θ, z)
        '''
        x = xp[0] * np.cos(xp[1])
        y = xp[0] * np.sin(xp[1])
        z = xp[2]
        
        return np.array([x, y, z])


class Sphere():
    '''
    球坐标系与直角坐标系转换
    
    
    Sphere transformation
    '''
    @classmethod
    def sph(self, x):
        '''
        球坐标正变换
        
        参数
        ----
        x：列表，(x, y, z)
        
        
        Parameters
        ----------
        x: list, (x, y, z)
        '''
        r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
        if r == 0:
            theta = 0
            phai = 0
        else:
            p = np.arccos(x[2] / r)
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


    @classmethod
    def isph(self, xp):
        '''
        球坐标逆变换
        
        参数
        ----
        xp：列表，(r, θ, φ)
        
        
        Parameters
        ----------
        xp: list, (r, θ, φ)
        '''
        x = xp[0] * np.sin(xp[2]) * np.cos(xp[1])
        y = xp[0] * np.sin(xp[2]) * np.sin(xp[1])
        z = xp[0] * np.cos(xp[2])
        
        return np.array([x, y, z])



class Rotate():
    '''
    平面直角坐标系的旋转变换
    
    参数
    ----
    theta：浮点数类型，坐标系绕原点逆时针旋转的角度
    
    
    Rotation transformation of plane rectangular coordinate system
    
    Parameter
    ---------
    theta: float, the angle that the coordinate system rotates counterclockwise about the origin
    '''
    def __init__(self, theta):
        self.__theta = theta
    
    
    def rot(self, x):
        '''
        旋转正变换
        
        参数
        ----
        x：列表，(x, y)
        
        
        Parameter
        ---------
        x: list, (x, y)
        '''
        xp = np.cos(self.__theta) * x[0] + np.sin(self.__theta) * x[1]
        yp = np.cos(self.__theta) * x[1] - np.sin(self.__theta) * x[0]
        
        return np.array([xp, yp])
    
    
    def irot(self, xp):
        '''
        旋转逆变换
        
        参数
        ----
        xp：列表，(x', y')
        
        
        Parameter
        ---------
        xp: list, (x', y')
        '''
        x = np.cos(self.__theta) * xp[0] - np.sin(self.__theta) * xp[1]
        y = np.cos(self.__theta) * xp[1] + np.sin(self.__theta) * xp[0]
        
        return np.array([x, y])