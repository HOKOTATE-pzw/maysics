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
        self.γ = 1/(1-(v/constant.c)**2)**0.5
    

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
        result = np.array([xp, yp, zp, tp])
        return result
    

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
        result = np.array([x, y, z, t])
        return result
    

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
        result = np.array([vxp, vyp, vzp])
        return result
    

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
        result = np.array([vxo, vyo, vzo])
        return result


class Polar():
    '''
    极坐标系与直角坐标系转换
    
    
    polar transformation
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
        result = np.array([r, theta])
        return result


    @classmethod
    def ipol(self, x):
        '''
        极坐标逆变换
        
        参数
        ----
        x：列表，(r, θ)
        
        
        Parameters
        ----------
        x: list, (r, θ)
        '''
        x = x[0] * np.cos(x[1])
        y = x[0] * np.sin(x[1])
        result = np.array([x, y])
        return result


class Cylinder():
    '''
    柱坐标系与直角坐标系转换
    
    cylinder transformation
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
        result = np.array([r, theta, z])
        return result


    @classmethod
    def icyl(self, x):
        '''
        柱坐标逆变换
        
        参数
        ----
        x：列表，(r, θ, z)
        
        
        Parameters
        ----------
        x: list, (r, θ, z)
        '''
        x = x[0] * np.cos(x[1])
        y = x[0] * np.sin(x[1])
        z = x[2]
        result = np.array([x, y, z])
        return result


class Sphere():
    '''
    球坐标系与直角坐标系转换
    
    sphere transformation
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
        result = np.array([r, theta, phai])
        return result


    @classmethod
    def isph(self, x):
        '''
        球坐标逆变换
        
        参数
        ----
        x：列表，(r, θ, φ)
        
        
        Parameters
        ----------
        x: list, (r, θ, φ)
        '''
        x = x[0] * np.sin(x[2]) * np.cos(x[1])
        y = x[0] * np.sin(x[2]) * np.sin(x[1])
        z = x[0] * np.cos(x[2])
        result = np.array([x, y, z])
        return result