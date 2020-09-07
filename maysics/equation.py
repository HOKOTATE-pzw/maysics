import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class Scatter():
    '''
    求解散点方程
    求解由一系列二维散点组成的曲线的交点
    
    参数
    ----
    x：一维或二维列表，仅支持以数作为自变量，不支持以列表作为自变量
    y：一维或二维列表
    acc：浮点数类型，可选，插值精度，默认0.1
    kind：浮点数类型或整型，可选，将插值类型指定为字符串('linear'、'nearest'、'zero'、'slinear'、'squardic'、'previous'、'next'，其中'zero'、'slinear'、'squared'和'cubic'表示零阶、一阶、二阶或三阶样条曲线插值；'previous'和'next'只返回点的上一个或下一个值)或作为一个整数指定要使用的样条曲线插值器的顺序。默认为'linear'
    xtol：浮点数类型，可选，横坐标误差，默认0.1
    ytol：浮点数类型，可选，纵坐标误差，默认0.01
    
    属性
    ----
    root：列表形式，解
    value：列表形式，解对应的函数值
    
    
    Solving the Equation of Dissolution Point
    Solving the intersection point of a curve composed of a series of two-dimensional scattered points
    
    Parameters
    ----------
    x: 1-D or 2-D list, only numbers are supported as arguments, lists are not
    y: 1-D or 2-D list
    acc: float, callable, interpolation accuracy, default=0.1
    kind: float or int, callable, specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. default='linear'
    xtol: float, callable, abscissa error, default=0.1
    ytol: float, callable, ordinate error, default=0.01
    
    Attributes
    ----------
    root: list, solution
    value: list, the function values of the solutions
    '''
    def __init__(self, x, y, acc=0.1, kind='linear',xtol=0.1, ytol=0.01):
        self.x = x
        self.y = y
        
        #判断输入列表维度
        try:
            x[0][0]
        except (TypeError, IndexError):
            self.x = [x]
            self.y = [y]
            zero = True
        
        min_list = []
        max_list = []
        for i in self.x:
            min_list.append(min(i))
            max_list.append(max(i))
        self.__min_x = max(min_list)
        self.__max_x = min(max_list)
        self.x_new = np.arange(self.__min_x, self.__max_x, acc)
        
        func_list = []
        self.y_new = []
        for i in range(len(self.x)):
            func = interp1d(self.x[i], self.y[i], kind)
            func_list.append(func)
            self.y_new.append(func(self.x_new))
        
        if zero:
            self.y_new.append(np.zeros(len(self.x_new)))
        
        self.y_new = np.array(self.y_new)
        self.root = []
        self.value = []
        
        #构造最小范围的散点
        x_scatter = []
        y_scatter = []
        for i in range(len(self.x)):
            x_scatter.append([])
            y_scatter.append([])
            for j in range(len(self.x[i])):
                if self.__min_x <= self.x[i][j] <= self.__max_x:
                    x_scatter[i].append(self.x[i][j])
                    y_scatter[i].append(self.y[i][j])
        self.x = x_scatter
        self.y = y_scatter
        
        x_buffer = []
        y_buffer = []
        for j in range(self.y_new.shape[1]):
            delta = []
            for i in range(self.y_new.shape[0]):
                delta.append(self.y_new[i, j])
            mean_delta = np.mean(delta)
            delta = max(delta) - min(delta)
            if delta <= ytol:
                x_buffer.append(self.x_new[j])
                y_buffer.append(mean_delta)
                if x_buffer[-1] - x_buffer[0] > xtol:
                    self.root.append(np.mean(x_buffer[:-1]))
                    x_buffer = [x_buffer[-1]]
                    self.value.append(np.mean(y_buffer[:-1]))
                    y_buffer = [y_buffer[-1]]
                
            if j == self.y_new.shape[1] - 1:
                self.root.append(np.mean(x_buffer))
                self.value.append(np.mean(y_buffer))
    
    
    def __image_process(self, scatter):
        for i in range(len(self.x)):
            plt.plot(self.x_new, self.y_new[i])
            if scatter:
                plt.plot(self.x[i], self.y[i], 'o')
        plt.plot(self.root, self.value, 'ro')
        
        
    def show(self, scatter=False):
        '''
        作图并显示
        
        参数
        ---
        scatter：布尔类型，True表示显示原散点，False表示不显示
        
        
        Display the image
        
        Parameter
        ---------
        scatter: bool, True means display the origin scatter, False means not
        '''
        Scatter.__image_process(self, scatter=scatter)
        plt.show()
    
    
    def savefig(self, filename, scatter=False):
        '''
        作图并保存
        
        参数
        ----
        filename：字符串类型，保存的文件名
        scatter：布尔类型，True表示显示原散点，False表示不显示
        
        
        Save the image
        
        Parameters
        ----------
        filename: str, file name
        scatter: bool, True means display the origin scatter, False means not
        '''
        Scatter.__image_process(self, scatter=scatter)
        plt.savefig(filename)