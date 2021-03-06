'''
本模块是额外工具箱

This module is extra Utils
'''
import numpy as np
from matplotlib import pyplot as plt
from urllib import request
from lxml import etree
from urllib import parse
import string


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
    
    返回
    ----
    元组形式，(筛选到的时间(列表), 筛选到的时间索引(列表))
    
    
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
    
    Return
    ------
    tuple, (filtered time(list), index of the filtered time(list))
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
    
    返回
    ----
    元组形式，(筛选到的时间(列表), 筛选到的时间索引(列表))
    
    
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
    
    Return
    ------
    tuple, (filtered time(list), index of the filtered time(list))
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
    
    返回
    ----
    元组形式，(筛选到的时间(列表), 筛选到的时间索引(列表))
    
    
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
    
    Return
    ------
    tuple, (filtered time(list), index of the filtered time(list))
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
    
    返回
    ----
    元组形式，(筛选到的时间(列表), 筛选到的时间索引(列表))
    
    
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
    
    Return
    ------
    tuple, (filtered time(list), index of the filtered time(list))
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