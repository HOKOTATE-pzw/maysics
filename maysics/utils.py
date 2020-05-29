'''
本模块用于预处理数据

This module is used for preprocessing data
'''

import numpy as np


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
    index：列表形式，需要进行标准化的列的索引，默认为全部
    mean：布尔类型，是否将均值调整为0
    var：布尔类型，是否将方差调整为1
    
    
    Standardize data
    z = (x - u) / s
    z: new data;  x: origin data;  u: mean value;  s: variance
    if data in one column are the same(s=0), data in this column will be turned to 0
    
    Parameters
    ----------
    data: 2-D ndarray
    index: list, index of columns need to be standardized, defalut to all
    mean: bool, if adjust the mean value to 0
    var: bool, if adjust the variance to 0
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