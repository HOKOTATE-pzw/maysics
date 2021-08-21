'''
本模块用于数字图像处理

This module is used for digital image process
'''
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def save(data, filename):
    '''
    将张量数据保存为图片
    
    参数
    ----
    data：列表类型，张量数据，列表的元素要求是整型
    filename：字符串类型，文件名
    
    
    Save the tensor as a picture
    
    Parameters
    ----------
    data: list, tensor, the elements of the data are required to be int
    filename: str, file name
    '''
    data = np.array(data, dtype='uint8')
    image = Image.fromarray(data)
    image.save(filename)


def load(filename, dtype='uint8'):
    '''
    将图片转换为张量数据
    
    参数
    ----
    filename：字符串类型，文件名
    dtype：可选，输出图像数据类型，默认为'uint8'
    
    返回
    ----
    ndarray，图片的张量数据
    
    
    Transform the picture into tensor
    
    Parameters
    ----------
    filename: str, file name
    dtype: callable, data format of output image, default='uint8'
    
    Return
    ------
    ndarray, the tensor of the picture
    '''
    x = Image.open(filename)
    return np.array(x, dtype=dtype)


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
    data: 2-D or 3-D ndarray, tensor of image
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2-D ndarray, grey image
    '''
    data = np.array(data, dtype=float)
    data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114
    data[data>255]=255
    data = np.around(data)
    return np.array(data, dtype=dtype)


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
    data: 2-D or 3-D ndarray, tensor of image
    
    Return
    ------
    1-D or 2-D ndarray, data of hitogram
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
    data: 2-D or 3-D ndarray, tensor of image
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


def linear_enhancement(data, begin, end, k, center=None, dtype=float):
    '''
    线性增强
    
    参数
    ----
    data：二维或三维ndarray，图像的张量数据
    begin：整型，线性增强区域的起点
    end：整型，线性增强区域的终点
    k：数类型，线性增强区域的斜率
    center：线性增强区域的不变点
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    二维或三维ndarray，图像的张量数据
    
    
    Linear Enhancement
    
    Parameters
    ----------
    data: 2-D or 3-D ndarray, tensor of image
    begin: starting point of linear enhancement region
    end: end point of linear enhancement region
    k: slope of linear enhancement region
    center: invariant point of linear enhancement region
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2-D or 3-D ndarray, tensor of image
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
    data: 2-D or 3-D ndarray, tensor of image
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2-D or 3-D ndarray, tensor of image
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


def laplace(data, mode=8, dtype=float):
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
    dtype：可选，输出图像数据的数据格式，默认为float
    
    返回
    ----
    二维ndarray，灰度图像
    
    
    Laplace Operator of Image
    
    Parameters
    ----------
    data: 2-D or 3-D ndarray, tensor of image
    mode: num, 4 and 8 are optional, represented two masks respectively:
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]]，
         and
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]]
        defualt=8
    dtype: callable, data format of output image, default=float
    
    Return
    ------
    2-D ndarray, grey image
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
    
    data_new = abs(data_new)
    data_new[data_new > 255] = 255
    return np.array(data_new, dtype=dtype)