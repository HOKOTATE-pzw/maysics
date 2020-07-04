'''
本模块用于数据预处理

This module is used for data preproccessing
'''
import numpy as np
from utils import Edis


def shuffle(*arg):
    '''
    以相同方法打乱多个序列或打乱一个序列
    
    返回值：一个ndarray
    
    
    shuffle multiple sequences in the same way or shuffle a sequences
    
    return: a ndarray
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
    
    
    split the data
    
    Parameters
    ----------
    data: data
    targets: targets
    train_size: float, callable, ratio of training set to total data, value range is (0, 1], default=0.75
    test_size: float, callable, ratio of test set to total data, value range is [0, 1)
    shuffle: bool, callable, 'True' will shuffle the data, 'False' will not, default = True
    random_state: int, callable, random seed
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
    
    返回值：训练集和验证集的元组
    
    
    Parameters
    ----------
    data: data
    targets: targets
    n: int, take the nth part as validation set, starting from 0
    k: int, callable, the number of k-fold, default = 5
    
    return: the tuple of training set and validation set
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
    
    
    Normalized data
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_new = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    Parameters
    ----------
    data: 2-D的ndarray数据
    index: list, callable, index of columns need to be standardized, defalut to all
    featur_range: tuple, callabel, final range of transformed data
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
    
    
    Making the moduli of data equal 1
    
    Parameters
    ----------
    data: 2-D的ndarray数据
    index: list, callable, index of columns need to be standardized, defalut to all
    '''
    data = np.array(data, dtype=np.float)
    
    if index:
        distance_list = data[:, index]**2
        distance_list = distance_list.sum(axis=1)**0.5
        distance_list = np.array([distance_list]).T
        distance_list[distance_list == 0] = 1
        print(distance_list)
        data[:, index] /= distance_list
    
    else:
        distance_list = data**2
        distance_list = distance_list.sum(axis=1)**0.5
        distance_list = np.array([distance_list]).T
        distance_list[distance_list == 0] = 1
        print(distance_list)
        data /= distance_list
    
    return data