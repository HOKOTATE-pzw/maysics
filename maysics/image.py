'''
本模块用于数字图像处理

This module is used for digital image process
'''
import numpy as np
from matplotlib import pyplot as plt


def l_convert(data, dtype=float):
    '''
    将RGB图像转换为灰度图像
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    二维ndarray，灰度图像
    
    
    Transform RGB image into grey image
    
    Parameters
    ----------
    data: 2D or 3D ndarray, tensor of image
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2D ndarray, grey image
    '''
    data = np.array(data, dtype=float)
    data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
    data[data>255]=255
    data = np.around(data)
    return np.array(data, dtype=dtype)


def hsv(pic, dtype=float):
    '''
    将RGB格式转换为HSV格式
    
    参数
    ----
    data：三维ndarray，图像的RGB张量数据
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    三维ndarray，HSV数据
    
    
    Transform RGB into HSV
    
    Parameters
    ----------
    data: 3D ndarray, RGB tensor of image
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    3D ndarray, HSV data
    '''
    pic = np.array(pic, dtype=float)
    v = pic.max(axis=2)
    med_v = pic.min(axis=2)
    s = np.zeros_like(v)
    s[v!=0] = 1 - med_v[v!=0] / v[v!=0]
    h = np.zeros_like(v)
    med_v = v - med_v
    
    for i in range(3):
        loc = np.where((v==pic[:, :, i]) & (med_v!=0))
        h[loc] = (pic[:, :, (i+1)%3][loc] - pic[:, :, (i+2)%3][loc]) * 60 / med_v[loc] + i * 120
    h[h<0] += 360
    
    pic = np.array([h, s, v], dtype=dtype)
    return pic.transpose(1, 2, 0)


def ihsv(pic, dtype=float):
    '''
    将HSV格式转换为RGB格式
    
    参数
    ----
    data：三维ndarray，图像的HSV张量数据
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    三维ndarray，HSV数据
    
    
    Transform HSV into RGB
    
    Parameters
    ----------
    data: 3D ndarray, HSV tensor of image
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    3D ndarray, RGB data
    '''
    pic = np.array(pic, dtype=float)
    h = pic[:, :, 0] / 60
    f = h.copy()
    h = h.astype(int) % 6
    f -= h
    v = [pic[:, :, 2] * (1 - pic[:, :, 1]),
         pic[:, :, 2] * (1 - (1 - f) * pic[:, :, 1]),
         pic[:, :, 2],
         pic[:, :, 2] * (1 - f * pic[:, :, 1])]
    
    # 初始化
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    # 分别处理h为偶数和奇数时的rgb
    for i in range(3):
        loc = np.where(h==2 * i)
        r[loc] = v[(i+2)%3][loc]
        g[loc] = v[(i+1)%3][loc]
        b[loc] = v[i%3][loc]
    del v[1]
    for i in range(3):
        loc = np.where(h==2 * i + 1)
        r[loc] = v[(i+2)%3][loc]
        g[loc] = v[(i+1)%3][loc]
        b[loc] = v[i%3][loc]
    
    pic = np.array([r, g, b], dtype=dtype)
    return pic.transpose(1, 2, 0)


def hist(data):
    '''
    图像的直方图
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    
    返回
    ----
    一维或二维ndarray，图像的直方图数据
    
    
    Histogram of Image
    
    Parameter
    ---------
    data: 2D or 3D ndarray, tensor of image
    
    Return
    ------
    1D or 2D ndarray, data of hitogram
    '''
    data = np.array(data)
    if len(data.shape) == 2:
        result = []
        for i in range(256):
            result.append(data[data==i].shape[0])
    
    elif len(data.shape) == 3:
        result = [[],[],[]]
        for j in range(3):
            data_copy = data[:, :, j]
            for i in range(256):
                result[j].append(data_copy[data_copy==i].shape[0])
    
    return np.array(result)


def hist_graph(data, mode=1, save=False):
    '''
    绘制图像的直方图
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    mode：数类型，可选1和2，1代表折线图，2代表直方图，默认为1
    save：字符串类型或布尔类型，若为字符串类型，表示保存图像为文件，若为False，则表示显示图像，默认为False
    
    
    Display the Histogram of Image
    
    Parameters
    ----------
    data: 2D or 3D ndarray, tensor of image
    mode: num, 1 and 2 are optional, 1 means line graph, 2 means histogram
    save: str or bool, str type means save graph as file, False means display graph, default=False
    '''
    data = hist(data)
    x = np.arange(0, 256, 1)
    if mode == 1 or mode == 'plot':
        if len(data.shape) == 1:
            plt.plot(x, data)
        
        elif len(data.shape) == 2:
            plt.plot(x, data[0], color='r')
            plt.plot(x, data[1], color='g')
            plt.plot(x, data[2], color='b')
    
    elif mode == 2 or mode == 'bar':
        if len(data.shape) == 1:
            plt.bar(x, data, width=0.5)
        
        elif len(data.shape) == 2:
            fig = plt.figure()
            colors = ['r', 'g', 'b']
            for i in range(3):
                ax = fig.add_subplot(3, 1, i+1)
                ax.bar(x, data[i], width=0.5, color=colors[i])
    
    plt.tight_layout()
    if save is False:
        plt.show()
    else:
        plt.savefig(save)


def _pre_linear_enhancement(data, begin, end, k, center):
    if not center:
        center = (begin + end) * 0.5
    inter1 = k * (begin - center) + center
    inter2 = k * (end - center) + center
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if begin <= data[i, j] <= end:
                data[i, j] = k * (data[i, j] - center) + center
            elif data[i, j] < begin:
                data[i, j] = inter1 * data[i, j] / begin
            elif data[i, j] > end:
                data[i, j] = (255 - inter2) * (data[i, j] - end) / (255 - end) + inter2
    
    return data


def linear_enhance(data, begin, end, k, center=None, dtype=float):
    '''
    线性增强
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    begin：整型，线性增强区域的起点
    end：整型，线性增强区域的终点
    k：数类型，线性增强区域的斜率
    center：线性增强区域的不变点，默认为begin与end的中点
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    二维或三维ndarray，图像的张量数据
    
    
    Linear Enhancement
    
    Parameters
    ----------
    data: 2D or 3D ndarray, tensor of image
    begin: starting point of linear enhancement region
    end: end point of linear enhancement region
    k: slope of linear enhancement region
    center: invariant point of linear enhancement region, the default is the midpoint between begin and end
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2D or 3D ndarray, tensor of image
    '''
    data = np.array(data, dtype=float)
    if len(data.shape) == 2:
        data_new = _pre_linear_enhancement(data, begin, end, k, center)
    elif len(data.shape) == 3:
        data_new = []
        data = data.transpose(2, 0, 1)
        for i in data:
            data_new.append(_pre_linear_enhancement(i, begin, end, k, center))
        data_new = np.array(data_new).transpose(1, 2, 0)
    
    data_new = np.around(data_new)
    data_new[data_new > 255] = 255
    data_new[data_new < 0] = 0
    return np.array(data_new, dtype=dtype)


def hist_equa(data, dtype=float):
    '''
    直方图均衡
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    二维或三维ndarray，图像的张量数据
    
    
    Parameters
    ----------
    data: 2D or 3D ndarray, tensor of image
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2D or 3D ndarray, tensor of image
    '''
    data = np.array(data, dtype=float)
    data_copy = data.copy()
    if len(data.shape) == 3:
        data_copy = data_copy[:, :, 0] * 0.299 + data_copy[:, :, 1] * 0.587 + data_copy[:, :, 2] * 0.114
        data_copy = np.around(data_copy)
    
    N = np.prod(data_copy.shape)
    grey = []
    for i in range(256):
        grey.append(data_copy[data_copy==i].shape[0])
        data[data==i] = sum(grey) * 255 / N
    
    data = np.around(data)
    return np.array(data, dtype=dtype)


def laplace(data, mode=8, strenth=1, dtype=float):
    '''
    图像的拉普拉斯算子
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    mode：数类型，可选4和8，分别表示两种掩膜：
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]]，
         和
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]]
        默认为8
    strenth：数类型，可选，对比的增强倍率，默认为1
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    二维ndarray，灰度图像
    
    
    Laplace Operator of Image
    
    Parameters
    ----------
    data: 2D or 3D ndarray, tensor of image
    mode: num, 4 and 8 are optional, represented two masks respectively:
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]]，
         and
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]]
        defualt=8
    strenth: num, callable, contrast enhancement magnification, default=1
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2D ndarray, grey image
    '''
    data = np.array(data, dtype=float)
    if len(data.shape) == 3:
        data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
    
    data_copy = data.copy()
    data_new = np.zeros_like(data)
    
    data_copy = np.hstack((data_copy[:, 0:1], data_copy[:, 0:-1]))
    data_new += data - data_copy
    data_copy = data
    
    data_copy = np.hstack((data_copy[:, 0:-1], data_copy[:, -1:]))
    data_new += data - data_copy
    data_copy = data
    
    data_copy = np.vstack((data_copy[0:1, :], data_copy[0:-1, :]))
    data_new += data - data_copy
    data_copy = data
    
    data_copy = np.vstack((data_copy[0:-1, :], data_copy[-1:, :]))
    data_new += data - data_copy
    data_copy = data
    
    if mode == 8:
        data_copy = np.hstack((data_copy[1:, 0:1], data_copy[:-1, :]))
        data_copy = np.vstack((data[0:1, :], data_copy[:, :-1]))
        data_new += data - data_copy
        data_copy = data
        
        data_copy = np.hstack((data_copy[:-1, 0:1], data_copy[1:, :]))
        data_copy = np.vstack((data_copy[:, :-1], data[-2:-1, :]))
        data_new += data - data_copy
        data_copy = data
        
        data_copy = np.hstack((data_copy[:-1, :], data_copy[1:, -2:-1]))
        data_copy = np.vstack((data[0:1, :], data_copy[:, 1:]))
        data_new += data - data_copy
        data_copy = data
        
        data_copy = np.hstack((data_copy[1:, :], data_copy[:-1, -2:-1]))
        data_copy = np.vstack((data_copy[:, 1:], data[-2:-1, :]))
        data_new += data - data_copy
        data_copy = data
    
    data_new = abs(data_new) * strenth
    data_new[data_new > 255] = 255
    return np.around(data_new).astype(dtype)


def saturate(data, scale=1, param={}, dtype=float):
    '''
    调整图片饱和度
    
    参数
    ----
    data：三维ndarray，图像的张量数据
    scale：数或函数类型，可选，当scale为数类型时，表示饱和度调整为S*scale；当scale为函数类型时，表示饱和度调整为scale(S)，默认为1
    param：字典类型，可选，当scale为函数类型且有其他非默认参数时，需输入以参数名为键，参数值为值的字典，默认为空字典
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    三维ndarray，图像
    
    
    Adjust the saturation
    
    Parameters
    ----------
    data: 3D ndarray, tensor of image
    scale: num or function, callable, when scale is num, it means adjust the saturation to S*scale; while it's function, it means adjust the saturation to scale(S), default=1
    param: dict, callable, When step is function and has other non-default parameters, "param" needs to be input a dictionary with parm_name as key and param_value as value, default={}
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    3D ndarray, image
    '''
    data = np.array(data, dtype=float)
    v = data.max(axis=2)
    med_v = data.min(axis=2)
    s = np.zeros_like(v)
    s[v!=0] = 1 - med_v[v!=0] / v[v!=0]
    h = np.zeros_like(v)
    med_v = v - med_v
    
    for i in range(3):
        loc = np.where((v==data[:, :, i]) & (med_v!=0))
        h[loc] = (data[:, :, (i+1)%3][loc] - data[:, :, (i+2)%3][loc]) / med_v[loc] + i * 2
    h[h<0] += 6
    
    # 调整饱和度s
    if type(scale).__name__ != 'function':
        s *= scale
    else:
        s = scale(s, **param)
    s[s>1] = 1
    
    f = h.copy()
    h = h.astype(int) % 6
    f -= h
    v = [v * (1 - s), v * (1 - (1 - f) * s), v, v * (1 - f * s)]
    
    # 初始化
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    # 分别处理h为偶数和奇数时的rgb
    for i in range(3):
        loc = np.where(h==2 * i)
        r[loc] = v[(i+2)%3][loc]
        g[loc] = v[(i+1)%3][loc]
        b[loc] = v[i%3][loc]
    del v[1]
    for i in range(3):
        loc = np.where(h==2 * i + 1)
        r[loc] = v[(i+2)%3][loc]
        g[loc] = v[(i+1)%3][loc]
        b[loc] = v[i%3][loc]
    
    data = np.array([r,g,b], dtype=dtype)
    return data.transpose(1, 2, 0)