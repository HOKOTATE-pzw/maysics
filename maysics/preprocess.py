'''
本模块用于数据预处理

This module is used for data preproccessing
'''
import numpy as np
from maysics.utils import e_distances
from matplotlib import pyplot as plt


def pad(seq, maxlen=None, value=0, padding='pre', dtype='int32'):
    '''
    填充二维列表，使得每行长度都为maxlen
    
    参数
    ----
    seq：二维列表，需要填充的对象
    maxlen：整型，可选，每行的最大长度，默认为原二维列表最大的长度
    value：数类型，可选，填充值，默认为0
    padding：字符串类型，可选，填充位置，'pre'代表从前面填充，'post'代表从后面填充，默认为'pre'
    dtype：可选，输出的元素类型，默认为'int32'
    
    返回
    ----
    二维ndarray
    
    
    Pad the 2-D list so that every row is 'maxlen' in length
    
    Parameters
    ----------
    seq: 2-D list, objects that need to be padded
    maxlen: int, callable, the maximum length of each row, default = the maximum length of the original 2-D list
    value: num, callable, padding value, default=0
    padding: str, callable, padding location, 'pre' means padding from the front and 'post' from the back, default='pre'
    dtype: callable, the element type of the output, default='int32'
    
    Return
    ------
    2-D ndarray
    '''
    seq = list(seq)
    if not maxlen:
        maxlen = 0
        for i in seq:
            if len(i) > maxlen:
                maxlen = len(i)
    
    if padding == 'pre':
        for i in range(len(seq)):
            if maxlen > len(seq[i]):
                seq[i] = [value] * (maxlen - len(seq[i])) + seq[i]
            elif maxlen < len(seq[i]):
                seq[i] = seq[i][-1 * maxlen:]
    
    elif padding == 'post':
        for i in range(len(seq)):
            if maxlen > len(seq[i]):
                seq[i] += [value] * (maxlen - len(seq[i]))
            elif maxlen < len(seq[i]):
                seq[i] = seq[i][:maxlen]
    
    return np.array(seq, dtype=dtype)


def shuffle(*arg):
    '''
    以相同方法打乱多个序列或打乱一个序列
    
    返回
    ----
    一个ndarray
    
    
    shuffle multiple sequences in the same way or shuffle a sequences
    
    Return
    ------
    a ndarray
    '''
    state = np.random.get_state()
    a_new_list = []
    for li in arg:
        np.random.set_state(state)
        np.random.shuffle(li)
        a_new_list.append(li)
    return np.array(a_new_list)


def data_split(data, targets, train_size=None, test_size=None, shuffle=True, random_state=None):
    '''
    分离数据
    
    参数
    ----
    data：数据
    targets：指标
    train_size：浮点数类型，可选，训练集占总数据量的比，取值范围为(0, 1]，默认为0.75
    test_size：浮点数类型，可选，测试集占总数据量的比，取值范围为[0, 1)，当train_size被定义时，该参数无效
    shuffle：布尔类型，可选，True表示打乱数据，False表示不打乱数据，默认为True
    random_state：整型，可选，随机种子
    
    返回
    ----
    元组，(数据测试集, 指标测试集, 数据验证集, 指标验证集)
    
    
    split the data
    
    Parameters
    ----------
    data: data
    targets: targets
    train_size: float, callable, ratio of training set to total data, value range is (0, 1], default=0.75
    test_size: float, callable, ratio of test set to total data, value range is [0, 1)
    shuffle: bool, callable, 'True' will shuffle the data, 'False' will not, default = True
    random_state: int, callable, random seed
    
    Return
    ------
    tuple, (train_data, train_target, validation_data, validation_target)
    '''
    if not (train_size or test_size):
        train_size = 0.75
    elif test_size:
        train_size = 1 - test_size
    
    if train_size <= 0 or train_size > 1:
        raise Exception("'train_size' should be in (0, 1], 'test_size' should be in [0, 1)")
    
    if shuffle:
        np.random.seed(random_state)
        
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(targets)
        
    num_of_data = len(data)
    train_data = data[:int(num_of_data * train_size)]
    train_target = targets[:int(num_of_data * train_size)]
    validation_data = data[int(num_of_data * train_size):]
    validation_target = targets[int(num_of_data * train_size):]
    
    return train_data, train_target, validation_data, validation_target


def kfold(data, targets, n, k=5):
    '''
    参数
    ----
    data：数据
    targets：指标
    n：整型，表示将第n折作为验证集，从0开始
    k：整型，可选，k折验证的折叠数，默认k=5
    
    返回
    ----
    元组，(数据测试集, 指标测试集, 数据验证集, 指标验证集)
    
    
    Parameters
    ----------
    data: data
    targets: targets
    n: int, take the nth part as validation set, starting from 0
    k: int, callable, the number of k-fold, default = 5
    
    Return
    ------
    tuple, (train_data, train_target, validation_data, validation_target)
    '''
    num_validation_samples = len(data) // k
    
    validation_data = data[num_validation_samples * n:
                           num_validation_samples * (n + 1)]
    validation_targets = targets[num_validation_samples * n:
                                 num_validation_samples * (n + 1)]
    
    train_data = np.concatenate([data[: num_validation_samples * n],
                                 data[num_validation_samples * (n + 1):]])
    train_targets = np.concatenate([targets[: num_validation_samples * n],
                                    targets[num_validation_samples * (n + 1):]])
    
    return train_data, train_targets, validation_data, validation_targets


def standard(data, index=None, mean=True, var=True):
    '''
    标准化数据
    z = (x - u) / s
    z：新数据；  x：原数据；  u：均值；  s：方差
    如果某一列数据完全相同（即方差s=0），则该列数据全部归零
    
    参数
    ----
    data：2-D的ndarray数据
    index：列表形式，可选，需要进行标准化的列的索引，默认为全部
    mean：布尔类型，可选，是否将均值调整为0
    var：布尔类型，可选，是否将方差调整为1
    
    返回
    ----
    2-D ndarray
    
    
    Standardize data
    z = (x - u) / s
    z: new data;  x: origin data;  u: mean value;  s: variance
    if data in one column are the same(s=0), data in this column will be turned to 0
    
    Parameters
    ----------
    data: 2-D ndarray
    index: list, callable, index of columns need to be standardized, defalut to all
    mean: bool, callable, if adjust the mean value to 0
    var: bool, callable, if adjust the variance to 0
    
    Return
    ------
    2-D ndarray
    '''
    data=np.array(data, dtype=np.float)
    
    if index:
        if mean:
            mean = data[:, index].mean(axis=0)
        else:
            mean = 0.0
        data[:, index] -= mean
        
        if var:
            std = data[:, index].std(axis=0)
            std_zero_indices = np.nonzero(std == 0)
            std[std==0] = 1.0
            data[:, index] /= std
            if list(std_zero_indices[0]):
                for i in std_zero_indices[0]:
                    data[:, index][:, i] *= 0
    else:
        if mean:
            mean = data.mean(axis=0)
        else:
            mean = 0.0
        data -= mean
        
        if var:
            std = data.std(axis=0)
            std_zero_indices = np.nonzero(std == 0)
            std[std==0] = 1.0
            data /= std
            if list(std_zero_indices[0]):
                for i in std_zero_indices[0]:
                    data[:, i] *= 0
    
    return data


def minmax(data, index=None, feature_range=(0, 1)):
    '''
    归一化数据
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_new = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    参数
    ----
    data：2-D的ndarray数据
    index：列表形式，可选，需要进行标准化的列的索引，默认为全部
    feature_range：元组形式，可选，需要转换的范围，默认为(0, 1)
    
    返回
    ----
    2-D ndarray
    
    
    Normalized data
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_new = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    Parameters
    ----------
    data: 2-D的ndarray数据
    index: list, callable, index of columns need to be standardized, defalut to all
    feature_range: tuple, callabel, final range of transformed data
    
    Return
    ------
    2-D ndarray
    '''
    data=np.array(data, dtype=np.float)
    
    if index:
        min_data = data[:, index].min(axis=0)
        length = data[:, index].max(axis=0) - min_data
        data[:, index] = (data[:, index] - min_data) / length
        data[:, index] = data[:, index] * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    else:
        min_data = data.min(axis=0)
        length = data.max(axis=0) - min_data
        data = (data - min_data) / length
        data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return data


def normalizer(data, index=None):
    '''
    使每个数据的模为1
    
    参数
    ----
    data：2-D的ndarray数据
    index：列表形式，可选，需要进行标准化的列的索引，默认为全部
    
    返回
    ----
    2-D ndarray
    
    
    Making the moduli of data equal 1
    
    Parameters
    ----------
    data: 2-D的ndarray数据
    index: list, callable, index of columns need to be standardized, defalut to all
    
    Return
    ------
    2-D ndarray
    '''
    data = np.array(data, dtype=np.float)
    
    if index:
        distance_list = e_distances(data[:, index])
        distance_list[distance_list == 0] = 1
        print(distance_list)
        data[:, index] /= distance_list
    
    else:
        distance_list = e_distances(data)
        distance_list[distance_list == 0] = 1
        print(distance_list)
        data /= distance_list
    
    return data



class Standard():
    '''
    标准化数据
    以一组数据为标准，将其他数据按照该组数据的方差和均值进行标准化
    z = (x - u) / s
    z：新数据；  x：原数据；  u：均值；  s：方差
    如果某一列数据完全相同（即方差s=0），则该列数据全部归零
    
    参数
    ----
    data：2-D的ndarray数据
    mean：布尔类型，可选，是否将均值调整为0
    var：布尔类型，可选，是否将方差调整为1
    
    属性
    ----
    mean：1-D的ndarray，数据的均值
    var：1-D的ndarray，数据的方差
    
    
    Standardize data
    Take a group of data as the standard, other data will be standardized according to the variance and mean value of this group of data
    z = (x - u) / s
    z: new data;  x: origin data;  u: mean value;  s: variance
    if data in one column are the same(s=0), data in this column will be turned to 0
    
    Parameters
    ----------
    data: 2-D ndarray
    mean: bool, callable, if adjust the mean value to 0
    var: bool, callable, if adjust the variance to 0
    
    Attributes
    ----------
    mean: 1-D ndarray, mean values of the data
    var: 1-D ndarray, variances of the data
    '''
    def __init__(self, data, mean=True, var=True):
        data=np.array(data, dtype=np.float)
        
        if mean:
            self.mean = data.mean(axis=0)
        else:
            self.mean = False
        
        if var:
            self.var = data.var(axis=0)
            std_zero_indices = np.nonzero(self.var == 0)
            self.var[self.var==0] = 1.0
            self.__std_zero_indices = std_zero_indices[0]
        else:
            self.var = False
    
    
    def transform(self, data, index=None):
        '''
        参数
        ----
        data：2-D的ndarray数据
        index：列表形式，可选，需要进行标准化的列的索引，默认为全部
        
        返回
        ----
        2-D ndarray
        
        
        Parameters
        ----------
        data: 2-D ndarray
        index: list, callable, index of columns need to be standardized, defalut to all
        
        Return
        ------
        2-D ndarray
        '''
        data=np.array(data, dtype=np.float)
        if index:
            if type(self.mean).__name__ == 'ndarray':
                data[:, index] -= self.mean[index]
            
            if type(self.var).__name__ == 'ndarray':
                data[:, index] /= self.var[index]
        
        else:
            if type(self.mean).__name__ == 'ndarray':
                data -= self.mean
            
            if type(self.var).__name__ == 'ndarray':
                data /= self.var
        
        return data



class Minmax():
    '''
    归一化数据
    以一组数据为标准，将其他数据按照该组数据的最大值和最小值进行归一化
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_new = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    参数
    ----
    data：2-D的ndarray数据
    
    属性
    ----
    minimum：1-D ndarray，最小值
    maximum：1-D ndarray，最大值
    length：1-D ndarray，最大值 - 最小值
    
    
    Normalized data
    Take a group of data as the standard, other data will be normalized according to the minimum and maximum value of this group of data
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_new = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    Parameter
    ---------
    data: 2-D的ndarray数据
    
    Attributes
    ----------
    minimum: 1-D ndarray
    maximum: 1-D ndarray
    length: 1-D ndarray, maximum - minimum
    '''
    def __init__(self, data):
        data=np.array(data, dtype=np.float)
        
        self.minimum = data.min(axis=0)
        self.maximum = data.max(axis=0)
        self.length = self.maximum - self.minimum
    
    
    def transform(self, data, index=None, feature_range=(0, 1)):
        '''
        参数
        ----
        data：2-D的ndarray数据
        index：列表形式，可选，需要进行标准化的列的索引，默认为全部
        feature_range：元组形式，可选，需要转换的范围，默认为(0, 1)
        
        返回
        ----
        2-D ndarray
        
        
        Parameters
        ----------
        data: 2-D的ndarray数据
        index: list, callable, index of columns need to be standardized, defalut to all
        feature_range: tuple, callabel, final range of transformed data
        
        
        Return
        ------
        2-D ndarray
        '''
        data=np.array(data, dtype=np.float)
        if index:
            data[:, index] = (data[:, index] - self.minimum[index]) / self.length[index]
            data[:, index] = data[:, index] * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        else:
            data = (data - self.minimum) / self.length
            data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        return data



class RC():
    '''
    相关系数
    
    参数
    ----
    *arg：列表类型
    
    属性
    ----
    rc_mat：相关系数矩阵
    
    
    correlation coefficient
    
    Parameter
    ---------
    *arg: list
    
    Attribute
    ---------
    rc_mat: correlation coefficient matrix
    '''
    def __init__(self, *arg):
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
        作图并显示
        
        参数
        ----
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Display the image
        
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
        作图并保存
        
        参数
        ----
        filename：字符串形式，文件名
        index：列表形式，可选，各数组名称
        cmap：字符串形式，可选，颜色板，默认为'Blues'
        
        
        Save the image
        
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