'''
本模块是额外工具箱

This module is extra Utils
'''

import numpy as np
from matplotlib import pyplot as plt


def time_before(time_list, time, itself=False, sep=True):
    '''
    寻找特定时间
    在时间列表中寻找某个时间点之前的时间
    要求time参数中时间的表述为'xxxx/xx/xx'，分别对应年-月-日，/表示可以是任意符号
    如“1970年1月1日”应表述为'1970/01/01'或'1970.01.01'等
    
    参数
    ----
    time_list：列表类型，时间列表
    time：字符串类型，设定的时间点
    itself：布尔类型，寻找时是否包括设定的时间点本身
    sep：布尔类型，time_list中的元素是否有符号将年月日分开，默认为True
    
    
    Search special time
    search all then time in time list which are before the point you set
    the expression of time are required to be like 'xxxx/xx/xx', corresponding to year-month-date, / can be any symbol
    e.g.  "January 1st, 1970" should be turned to '1970/01/01' or '1970.01.01' etc
    
    Parameters
    ----------
    time_list: list, list of time
    time: str, the time point
    itself: bool, if include the time point itself when searching
    sep: bool, True means elements in time_list have symbols to seperate the year, month and day, default=True
    '''
    select_time = []
    select_index = []
    date = int(time[:4]+time[5:7]+time[8:10])
    
    for i in range(len(time_list)):
        judge_time = time_list[i]
        if sep:
            judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
        else:
            judge_num = int(judge_time)
        
        if date > judge_num:
            select_time.append(judge_time)
            select_index.append(i)
        
        if itself:
            if date == judge_num:
                select_time.append(judge_time)
                select_index.append(i)
    
    return select_time, select_index


def time_after(time_list, time, itself=False, sep=True):
    '''
    寻找特定时间
    在时间列表中寻找某个时间点之后的时间
    要求时间的表述为'xxxx/xx/xx'，分别对应年-月-日，/表示可以是任意符号
    如“1970年1月1日”应表述为'1970/01/01'或'1970.01.01'等
    
    参数
    ----
    time_list：列表类型，时间列表
    time：字符串类型，设定的时间点
    itself：布尔类型，寻找时是否包括设定的时间点本身
    sep：布尔类型，time_list中的元素是否有符号将年月日分开，默认为True
    
    
    Search special time
    search all then time in time list which are after the point you set
    the expression of time are required to be like 'xxxx/xx/xx', corresponding to year-month-date, / can be any symbol
    e.g.  "January 1st, 1970" should be turned to '1970/01/01' or '1970.01.01' etc
    
    Parameters
    ----------
    time_list: list, list of time
    time: str, the time point
    itself: bool, if include the time point itself when searching
    sep: bool, True means elements in time_list have symbols to seperate the year, month and day, default=True
    '''
    select_time = []
    select_index = []
    date = int(time[:4] + time[5:7] + time[8:10])
    
    for i in range(len(time_list)):
        judge_time = time_list[i]
        if sep:
            judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
        else:
            judge_num = int(judge_time)
        
        if judge_num > date:
            select_time.append(judge_time)
            select_index.append(i)
        
        if itself:
            if date == judge_num:
                select_time.append(judge_time)
                select_index.append(i)
    
    return select_time, select_index


def time_equal(time_list, time, sep=True, equal=True):
    '''
    寻找特定时间的索引
    在时间列表中寻找某个时间点
    要求时间的表述为'xxxx/xx/xx'，分别对应年-月-日，/表示可以是任意符号
    如“1970年1月1日”应表述为'1970/01/01'或'1970.01.01'等
    
    参数
    ----
    time_list：列表类型，时间列表
    time：列表类型，设定的时间点列表
    sep：布尔类型，time_list中的元素是否有符号将年月日分开，默认为True
    equal：布尔类型，True表示搜索与time相同的时间，False表示搜索与time不相同的时间
    
    
    Search index of special time
    search all then time in time list which are the same as the point you set
    the expression of time are required to be like 'xxxx/xx/xx', corresponding to year-month-date, / can be any symbol
    e.g.  "January 1st, 1970" should be turned to '1970/01/01' or '1970.01.01' etc
    
    Parameters
    ----------
    time_list: list, list of time
    time: list, the time point list
    sep: bool, True means elements in time_list have symbols to seperate the year, month and day, default=True
    equal: bool, True means searching the time in "time", False means searching the time not in "time"
    '''
    select_index = []
    select_time = []
    for i in range(len(time)):
        time[i] = int(time[i][:4] + time[i][5:7] + time[i][8:10])
    
    if equal:
        for i in range(len(time_list)):
            judge_time = time_list[i]
            if sep:
                judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
            else:
                judge_num = int(judge_time)
            
            if judge_num in time:
                select_index.append(i)
                select_time.append(judge_time)
    
    else:
        for i in range(len(time_list)):
            judge_time = time_list[i]
            if sep:
                judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
            else:
                judge_num = int(judge_time)
            
            if not judge_num in time:
                select_index.append(i)
                select_time.append(judge_time)
        
        return select_time, select_index


def time_between(time_list, begin, end, begin_itself=False, end_itself=False, sep=True):
    '''
    寻找特定时间
    在时间列表中寻找某个时间点之后的时间
    要求时间的表述为'xxxx/xx/xx'，分别对应年-月-日，/表示可以是任意符号
    如“1970年1月1日”应表述为'1970/01/01'或'1970.01.01'等
    
    参数
    ----
    time_list：列表类型，时间列表
    begin：字符串类型，设定的开始时间点
    end：字符串类型，设定的结束时间点
    begin_itself：布尔类型，寻找时是否包括设定的开始时间点本身
    end_itself：布尔类型，寻找时是否包括设定的结束时间点本身
    sep：布尔类型，time_list中的元素是否有符号将年月日分开，默认为True
    
    
    Search special time
    search all then time in time list which are after the point you set
    the expression of time are required to be like 'xxxx/xx/xx', corresponding to year-month-date, / can be any symbol
    e.g.  "January 1st, 1970" should be turned to '1970/01/01' or '1970.01.01' etc
    
    Parameters
    ----------
    time_list: list, list of time
    begin: str, the beginning time point
    end: str, the end time point
    bengin_itself: bool, if include the beginning time point itself when searching
    end_itself: bool, if include the end time point itself when searching
    sep: bool, True means elements in time_list have symbols to seperate the year, month and day, default=True
    '''
    select_time = []
    select_index = []
    
    # begin
    date_1 = int(begin[:4] + begin[5:7] + begin[8:10])
    
    # end
    date_2 = int(end[:4] + end[5:7] + end[8:10])
    
    if not begin_itself and not end_itself:
        for i in range(len(time_list)):
            judge_time = time_list[i]
            if sep:
                judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
            else:
                judge_num = int(judge_time)
            
            if date_1 < judge_num < date_2:
                select_time.append(judge_time)
                select_index.append(i)
    
    elif begin_itself and not end_itself:
        for i in range(len(time_list)):
            judge_time = time_list[i]
            if sep:
                judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
            else:
                judge_num = int(judge_time)
            
            if date_1 <= judge_num < date_2:
                select_time.append(judge_time)
                select_index.append(i)
    
    elif not begin_itself and end_itself:
        for i in range(len(time_list)):
            judge_time = time_list[i]
            if sep:
                judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
            else:
                judge_num = int(judge_time)
            
            if date_1 < judge_num <= date_2:
                select_time.append(judge_time)
                select_index.append(i)
    
    else:
        for i in range(len(time_list)):
            judge_time = time_list[i]
            if sep:
                judge_num = int(judge_time[:4] + judge_time[5:7] + judge_time[8:10])
            else:
                judge_num = int(judge_time)
            
            if date_1 <= judge_num <= date_2:
                select_time.append(judge_time)
                select_index.append(i)
    
    return select_time, select_index


def grid_net(*args):
    '''
    生成网格点
    将输入的列表遍历组合
    
    
    Generate grid
    traverse and combine the input list
    '''
    net = np.meshgrid(*args)
    for i in range(len(net)):
        net[i] = net[i].flatten()
    net = np.vstack(tuple(net)).T
    return net


class rc():
    '''
    相关系数
    
    
    correlation coefficient
    '''
    def fit(self, *arg):
        arg = np.array(arg, dtype=float)
        if len(arg.shape) != 2:
            raise Exception("Input list should be 1-D.")
        
        cov_mat = np.cov(arg)
        var_mat = np.diagonal(cov_mat)**0.5
        var_mat[var_mat == 0] = 1
        
        for i in range(cov_mat.shape[0]):
            cov_mat[i] /= var_mat[i]
            cov_mat[:, i] /= var_mat[i]
        
        self.rc_mat = cov_mat
    
    
    def show(self, index=None, cmap='Blues'):
        '''
        参数
        ----
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Parameters
        ----------
        index: list, callable, names of each array
        cmap: str, callable, color board, default='Blues'
        '''
        plt.matshow(self.rc_mat, cmap=cmap)
        plt.colorbar()
        if index:
            n_list = range(len(index))
            plt.xticks(n_list, index)
            plt.yticks(n_list, index)
        plt.show()
    
    
    def savefig(self, filename, index=None, cmap='Blues'):
        '''
        参数
        ----
        filename：字符串形式，文件名
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Parameters
        ----------
        filename: str, file name
        index: list, callable, names of each array
        cmap: str, callable, color board, default='Blues'
        '''
        plt.matshow(self.rc_mat, cmap=cmap)
        plt.colorbar()
        if index:
            n_list = range(len(self.rc_mat))
            plt.xticks(n_list, index)
            plt.yticks(n_list, index)
        plt.savefig(filename)



class Edis():
    '''
    欧式距离
    
    参数
    ----
    data：一维或二维列表，数据
    
    
    Euclidean distance
    
    Parameters
    ----------
    data: 1-D or 2-D list, data
    '''
    def __init__(self, data):
        self.__data = np.array(data, dtype=np.float)
        
        if len(self.__data.shape) < 3:
            self.__n = data.shape[-1]
        else:
            raise Exception("Parameter 'data' must be 1-D or 2-D.")
    
    
    @classmethod
    def distance(self, p1, p2):
        '''
        求某两个点之间的距离
        
        参数
        ----
        p1：一维数组，第一个点的位置
        p2：一维数组，第二个点的位置
        
        
        Calculate the distance between two points
        
        Parameters
        ----------
        p1: 1-D list, the location of the first point
        p2: 1-D list, the location of the second point
        '''
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        return sum((p1 - p2)**2)**0.5
    
    
    def distances(self, des='o'):
        '''
        求data到目标点距离
        
        参数
        ----
        des：字符串或一维数组，可选'o'或'O'(原点)、'mean'(均值点)及自定义数组，目标点坐标，默认为'o'
        
        
        Calculate the distance between data and destination
        
        Parameter
        ---------
        des: str or 1-D list, 'o' or 'O' (origin), 'mean' (mean point) and custom array are optional, the coordinate of destination, default='o'
        '''
        if des == 'o' or des == 'O':
            des = np.zeros(self.__n)
    
        elif des == 'mean':
            des = self.__data.mean(axis=0)
        
        else:
            des = np.array(des)
    
        self.__data -= des
        self.__data = self.__data**2
        result = self.__data.sum(axis=len(self.__data.shape)-1)
        result = result**0.5
        
        return result



class Mdis():
    '''
    马氏距离
    
    参数
    ----
    data：二维列表，数据
    
    
    Mahalanobis distance
    
    Parameters
    ----------
    data: 2-D list, data
    '''
    def __init__(self, data):
        self.__data = np.mat(data, dtype=np.float)
        self.__dataT = self.__data.T
        
        if len(self.__data.shape) != 2:
            raise Exception("Parameter 'data' must be 2-D.")
        
        self.__SI = np.mat(np.cov(self.__dataT)).I
    
    
    def distance(self, p1, p2):
        '''
        求某两个点之间的距离
        
        参数
        ----
        p1：一维或二维数组，第一个点的位置
        p2：一维或二维数组，第二个点的位置
        
        
        Calculate the distance between two points
        
        Parameters
        ----------
        p1: 1-D or 2-D list, the location of the first point
        p2: 1-D or 2-D list, the location of the second point
        '''
        p1 = np.mat(p1, dtype=np.float)
        p2 = np.mat(p2, dtype=np.float)
        result = (p1 - p2) * self.__SI * (p1 - p2).T
        
        return result[0, 0]**0.5
    
    
    def distances(self, des='o'):
        '''
        求data到目标点距离
        
        参数
        ----
        des：字符串或一维或二维数组，可选'o'或'O'(原点)、'mean'(均值点)及自定义数组，目标点坐标，默认为'o'
        
        
        Calculate the distance between data and destination
        
        Parameter
        ---------
        des: str or 1-D or 2-D list, 'o' or 'O' (origin), 'mean' (mean point) and custom array are optional, the coordinate of destination, default='o'
        '''
        if des == 'o' or des == 'O':
            des = np.zeros(self.__data.shape[-1])
        
        elif des == 'mean':
            des = self.__data.mean(axis=0)
        
        else:
            des = np.mat(des)
        
        return np.diag((self.__data - des) * self.__SI *(self.__dataT - des.T))**0.5