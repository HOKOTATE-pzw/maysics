'''
本模块是额外工具箱

This module is extra Utils
'''
import numpy as np
from urllib import request
from lxml import etree
from urllib import parse
import string
from matplotlib import pyplot as plt


def grid_net(*args):
    '''
    生成网格点
    将输入的列表遍历组合
    如：
    grid_net([a, b], [c, d]) 或 grid_net(*[[a, b], [c, d]])
    返回：array([[a, c], [b, c], [a, d], [b, d]])
    
    
    Generate grid
    traverse and combine the input list
    e.g.
    grid_net([a, b], [c, d]) or grid_net(*[[a, b], [c, d]])
    return: array([[a, c], [b, c], [a, d], [b, d]])
    '''
    net = np.meshgrid(*args)
    for i in range(len(net)):
        net[i] = net[i].flatten()
    net = np.vstack(tuple(net)).T
    return net


def e_distance(p1, p2):
    '''
    求某两个点之间的欧式距离
    
    参数
    ----
    p1：一维数组，第一个点的位置
    p2：一维数组，第二个点的位置
    
    返回
    ---
    浮点数类型，距离
    
    
    Calculate the Euclidean distance between two points
    
    Parameters
    ----------
    p1: 1-D list, the location of the first point
    p2: 1-D list, the location of the second point
    
    Return
    ------
    float, the distance
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return sum((p1 - p2)**2)**0.5


def e_distances(data, des='o'):
    '''
    求data到目标点的欧式距离
    
    参数
    ----
    data：一维或二维列表，数据
    des：字符串或一维数组，可选'o'或'O'(原点)、'mean'(均值点)及自定义数组，目标点坐标，默认为'o'
    
    返回
    ----
    ndarray类型，距离数组
    
    
    Calculate the Euclidean distances between data and destination
    
    Parameter
    ---------
    data: 1-D or 2-D list, data
    des: str or 1-D list, 'o' or 'O' (origin), 'mean' (mean point) and custom array are optional, the coordinate of destination, default='o'
    
    Return
    ------
    ndarray, the distances
    '''
    data = np.array(data, dtype=np.float)
    if len(data.shape) < 3:
        n = data.shape[-1]
    else:
        raise Exception("Parameter 'data' must be 1-D or 2-D.")

    if des == 'o' or des == 'O':
        des = np.zeros(n)

    elif des == 'mean':
        des = data.mean(axis=0)
        
    else:
        des = np.array(des)
    
    data = (data - des)**2
    result = data.sum(axis=len(data.shape)-1)
    result = result**0.5
    
    return result


def m_distance(data, p1, p2):
    '''
    求某两个点之间的马氏距离
    
    参数
    ----
    data：二维列表，数据
    p1：一维或二维数组，第一个点的位置
    p2：一维或二维数组，第二个点的位置
    
    返回
    ---
    浮点数类型，距离
    
    
    Calculate the Mahalanobis distance between two points
    
    Parameters
    ----------
    data: 2-D list, data
    p1: 1-D or 2-D list, the location of the first point
    p2: 1-D or 2-D list, the location of the second point
    
    Return
    ------
    float, the distance
    '''
    data = np.mat(data, dtype=np.float)
    dataT = data.T
    
    if len(data.shape) != 2:
        raise Exception("Parameter 'data' must be 2-D.")
    
    SI = np.mat(np.cov(dataT)).I

    p1 = np.mat(p1, dtype=np.float)
    p2 = np.mat(p2, dtype=np.float)
    result = (p1 - p2) * SI * (p1 - p2).T
    
    return result[0, 0]**0.5


def m_distances(data, des='o'):
    '''
    求data到目标点的马氏距离
    
    参数
    ----
    data：二维列表，数据
    des：字符串或一维或二维数组，可选'o'或'O'(原点)、'mean'(均值点)及自定义数组，目标点坐标，默认为'o'
    
    返回
    ----
    ndarray类型，距离数组
    
    
    Calculate the Mahalanobis distance between data and destination
    
    Parameter
    ---------
    data: 2-D list, data
    des: str or 1-D or 2-D list, 'o' or 'O' (origin), 'mean' (mean point) and custom array are optional, the coordinate of destination, default='o'
    
    Return
    ------
    ndarray, the distances
    '''
    data = np.mat(data, dtype=np.float)
    dataT = data.T
    
    if len(data.shape) != 2:
        raise Exception("Parameter 'data' must be 2-D.")
    
    SI = np.mat(np.cov(dataT)).I

    if des == 'o' or des == 'O':
        des = np.zeros(data.shape[-1])
    
    elif des == 'mean':
        des = data.mean(axis=0)
    
    else:
        des = np.mat(des)
    
    return np.diag((data - des) * SI *(dataT - des.T))**0.5


def discrete(x, y, color=None, label=None):
    '''
    绘制离散函数图像
    
    参数
    ----
    x：一维数组类型，自变量
    y：一维数组类型，因变量
    color：字符串类型，可选，颜色
    label：字符串类型，可选，标签
    
    
    Draw the graph of discrete function
    
    Parameters
    ----------
    x: 1-D array, independent variable
    y: 1-D array, dependent variable
    color: str, callable, color
    label: str, callable, label
    '''
    plt.scatter(x, y, color=color, label=label)
    zeros = np.zeros_like(y)
    if not color:
        plt.vlines(x, zeros, y)
    else:
        plt.vlines(x, zeros, y, color=color)


def circle(center=(0, 0), radius=1, angle_range=(0, 2*np.pi), acc=0.01, label=None):
    '''
    绘制一个圆
    
    参数
    ----
    center：元组类型，可选，圆心坐标，默认为(0, 0)
    radius：数类型，可选，半径，默认为1
    angle_range：元组类型，可选，绘制的角度范围，默认为(0, 2π)
    acc：浮点数类型，可选，绘制的精度，默认为0.01
    label：字符串类型，可选，标签，默认为None
    
    
    Draw a circle
    
    Parameters
    ----------
    center: tuple, callable, center coordinate, default=(0, 0)
    radius: num, callable, radius, default=1
    angle_range: tuple, callable, the range of angle to draw, default=(0, 2π)
    acc: float, callable, the accuracy of drawing, default=0.01
    label: str, callable, label, default=None
    '''
    theta = np.arange(*angle_range, acc)
    radius = radius * np.ones_like(theta)
    x = np.vstack((radius, theta)).T
    x = tf.ipolar(x)
    plt.plot(x[:, 0] + center[0], x[:, 1] + center[1])


class Crawler():
    '''
    用于简单的爬虫
    
    参数
    ----
    url：链接
    headers：字典类型，可选
    encoding：编码类型
    timeout：等待时间，单位为秒，默认为全局时间
    
    属性
    ----
    html：爬取的html文本
    
    
    Used for simple web crawlers
    
    Parameters
    ----------
    url: URL
    headers: dict, callable
    encoding: encode type
    timeout: waiting time, in seconds, default to global time
    
    Attribute
    ---------
    html: crawled html text
    '''
    def __init__(self, url, headers={}, encoding='utf-8', timeout=None):
        url = parse.quote(url, safe = string.printable)
        
        if url[:4] == 'http':
            self.__file = False
            r = request.Request(url, headers=headers)
            self.__response = request.urlopen(r, timeout=timeout)
            self.html = self.__response.read().decode(encoding)
        else:
            self.__file = True
            with open(url) as self.__response:
                self.html = self.__response.read()
    
    
    def getcode(self):
        '''
        获取http状态码
        
        返回
        ----
        整数类型或字符串类型，http状态码
        
        
        Get http status code
        
        Return
        ------
        int or str, http status code
        '''
        if not self.__file:
            return self.__response.getcode()
        else:
            return 'local'
    
    
    def xpath_find(self, nod):
        '''
        以XPath的方式查找节点
        
        参数
        ----
        nod：字符串类型，节点
        
        返回
        ----
        字符串类型，查找到的内容
        
        
        Find nodes with XPath
        
        Parameter
        ---------
        nod: str, nod
        
        Return
        ------
        str, found content
        '''
        html = etree.HTML(self.html)
        return html.xpath(nod)
    
    
    def easy_find(self, nod, search='text'):
        '''
        以列表形式查找节点
        
        参数
        ----
        nod：字符串或一维、二维列表形式，列表形式为[节点名, 匹配属性]，若仅用到一个节点且无需属性匹配，可用字符串；若用到多个节点，则使用二维列表
            例如：
            要查找<div> text </div>中的text，则可以
            nod='div'
            
            要查找<div, class="class-name"><a> text </a></div>中的text，则可以
            nod=[['div', 'class="class-name"'], [a]]
        search：要查找的内容，可以是属性和文本，默认为'text'
        
        返回
        ----
        字符串类型，查找到的内容
        
        
        Find nodes with list
        
        Parameters
        ----------
        nod: str, 1-D or 2-D list, the form of the list is [node name, matching attribute], if only one node is used and no attribute matching is required, string is available; if multiple nodes are used, then use 2-D list
            e.g.
            to find the text in <div> text </div>
            nod='div'
            
            to find the text in <div, class = "class name"><a> text </a></div>
            nod=[['div', 'class="class-name"'], [a]]
        search: the content to be searched, attribute or text, default='text'
        
        Return
        ------
        str, found content
        '''
        nod_list = ''
        
        if not nod:
            pass
        
        elif isinstance(nod, str):
            nod_list = '//' + nod
        
        elif not isinstance(nod[0], list):
            nod = [nod]
            for i in nod:
                if len(i) == 1:
                    nod_list = nod_list + '//' + i[0]
                else:
                    nod_list = nod_list + '//' + i[0] + '[@' + i[1] + ']'
        
        if search == 'text':
            nod_list += '//text()'
        
        else:
            nod_list = nod_list + '@' + search
        
        html = etree.HTML(self.html)
        return html.xpath(nod_list)


class A_P():
    '''
    将信号的频域表示分解为“幅度-频率”和“相位-频率”

    参数
    ----
    X：函数或一维数组形式，信号的频域表示


    Decompose frequency domain representation of signal into "amplitude-frequency" and "phase-frequency"

    Parameter
    ---------
    X: function or 1-D array, frequency domain representation of signal
    '''
    def __init__(self, X):
        self.X = X
    

    def fit(self, f):
        '''
        计算频率为f时的幅度和相位

        参数
        ----
        f：数或一维数组形式，频率

        返回
        ----
        元组形式，(幅度, 相位)


        Calculate the amplitude and phase at frequency f

        Parameter
        ---------
        f: num or 1-D array, frequency

        Return
        ------
        tuple, (amplitude, phase)
        '''
        f = np.array(f)
        if type(self.X).__name__ == 'function':
            result = self.X(f)
        else:
            X = np.array(self.X)
            result = self.X[f]
        result = np.array(result)
        return abs(result), np.arctan(result.imag / result.real)
    

    def __image_process(self, f, image_type):
        f = np.array(f)
        amplitude, phase = self.fit(f)
        fig = plt.figure()
        if image_type == 'C' or image_type == 'c':
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(f, amplitude)
            ax.set_title('amplitude')
            ax = fig.add_subplot(2, 1, 2)
            ax.plot(f, phase)
            ax.set_title('phase')
        elif image_type == 'D' or image_type == 'd':
            zeros_list = np.zeros(f.shape)
            ax = fig.add_subplot(2, 1, 1)
            ax.scatter(f, amplitude, marker='o', s=30, zorder=3)
            ax.vlines(f, zeros_list, amplitude)
            ax.set_title('amplitude')
            ax = fig.add_subplot(2, 1, 2)
            ax.scatter(f, phase, marker='o', s=30, zorder=3)
            ax.vlines(f, zeros_list, phase)
            ax.set_title('phase')
        plt.tight_layout()
    

    def show(self, f, image_type='c'):
        '''
        显示“幅度-频率”图和“相位-频率”图

        参数
        ----
        f：数或一维数组形式，频率
        image_type：字符串形式，可选'c'和'd'，'c'表示绘制连续图像，'d'表示绘制离散图像，默认为'd'


        Display "amplitude-frequency" and "phase-frequency" graphs

        Parameters
        ----------
        f: num or 1-D array, frequency
        image_type: str, 'c' and 'd' are callable, 'c' means drawing continuous image and 'd'means drawing discrete image, default='c'
        '''
        self.__image_process(f, image_type)
        plt.show()
    

    def savefig(self, filename, f, image_type='c'):
        '''
        储存“幅度-频率”图和“相位-频率”图

        参数
        ----
        filename：字符串形式，文件名
        f：数或一维数组形式，频率
        image_type：字符串形式，可选'C'和'D'，'C'表示绘制连续图像，'D'表示绘制离散图像，默认为'C'


        Save "amplitude-frequency" and "phase-frequency" graphs

        Parameters
        ----------
        filename: str, file name
        f: num or 1-D array, frequency
        image_type: str, 'c' and 'd' are callable, 'c' means drawing continuous image and 'd'means drawing discrete image, default='c'
        '''
        self.__image_process(f, image_type)
        plt.savefig(filename)