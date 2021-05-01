'''
本模块用于数据预处理

This module is used for data preproccessing
'''
import numpy as np
from maysics.utils import e_distances
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
from io import BytesIO
from lxml import etree
import base64
import math


def _rc(*arg):
    cov_mat = np.cov(arg)
    var_mat = np.diagonal(cov_mat)**0.5
    var_mat[var_mat == 0] = 1
    
    for i in range(cov_mat.shape[0]):
        cov_mat[i] /= var_mat[i]
        cov_mat[:, i] /= var_mat[i]
    
    return cov_mat


def _preview_process(data, value_round):
    '''
    预览处理
    '''
    data = np.array(data, dtype=np.float)
    
    name_list = ['平均值', '中位数', '方差', '标准差', '最大值', '最小值', '偏度', '峰度']
    value_list = []
    mean_ = data.mean(axis=0)
    value_list.append(np.round(mean_, value_round))
    value_list.append(np.round(np.median(data, axis=0), value_round))
    value_list.append(np.round(data.var(axis=0), value_round))
    value_list.append(np.round(data.std(axis=0), value_round))
    value_list.append(np.round(data.max(axis=0), value_round))
    value_list.append(np.round(data.min(axis=0), value_round))
    value_list.append(np.round(((data - mean_)**3).mean(axis=0), value_round))
    value_list.append(np.round(((data - mean_)**4).mean(axis=0), value_round))
    value_list = np.array(value_list).flatten()
    
    style = '''
    <style>
    table{
        border-collapse: collapse;
    }
    table, table tr td {
        border:1px solid #ccc;
    }
    table tr td{
        padding: 5px 10px;
    }
    </style>
    '''
    table = '<h2 style="padding-left:50px; border-top:1px solid #ccc">数值特征</h2>' + style + '<table align="center"><caption></caption>'
    for i in range(8):
        table += '<tr><td>' + name_list[i] + '</td>' + '<td>%s</td>' * data.shape[1] + '</tr>'
    table = '<h1 style="padding-left:50px;">数据信息</h1>' + table % tuple(value_list) + '</table>'
    
    
    data = data.T
    num = data.shape[0]
    plt.figure(figsize=(9, 3 * num))
    for i in range(num):
        q1, q2, q3 = np.percentile(data[i], [25, 50, 75])
        plt.scatter(mean_[i], i+1, marker='o', color='white', s=30, zorder=3)
        plt.hlines(i+1, q1, q3, color='k', linestyle='-', lw=1)
    ax = plt.violinplot(data.tolist(), showextrema=False, vert=False)
    plt.title('分布图')
    
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = 'data:image/png;base64,' + ims
    im1 = '<div align="center"><img src="%s"></div>' % imd
    im1 = '<br></br><h2 style="padding-left:50px; border-top:1px solid #ccc">密度分布</h2>' + im1
    
    cov_mat = _rc(*data)
    matrix = '<table border="0"><caption></caption>'
    
    for i in range(num):
        matrix += '<tr>' + '<td>%s</td>' * num + '</tr>'
    matrix = matrix % tuple(np.round(cov_mat.flatten(), value_round)) + '</table>'
    
    
    plt.figure(figsize=(8, 8))
    plt.matshow(cov_mat, fignum=0, cmap='Blues')
    plt.colorbar()
    plt.title('相关系数图')
    
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = 'data:image/png;base64,' + ims
    im2 = '<div style="display:flex;flex-direction:row;vertical-align:middle;justify-content:center;width:100%;height:80vh"><div style="margin:auto 0;white-space:pre-wrap;max-width:50%">'
    im2 = im2 +'相关矩阵：'+ matrix + '</div><img style="object-fit:contain;max-width:45%;max-height:80vh" src="{}"/></div>'.format(imd)
    im2 = '<br></br><h2 style="padding-left:50px; border-top:1px solid #ccc">相关性</h2>' + im2
    
    
    plt.figure(figsize=(2.5 * num, 2.5 * num))
    for i in range(num * num):
        ax = plt.subplot(num, num, i+1)
        ax.plot(data[i//num], data[i%num], 'o')
    
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    im3 = '<div align="center"><img src="%s"></div>' % imd
    im3 = '<br></br><h2 style="padding-left:50px; border-top:1px solid #ccc">散点关系</h2>' + im3
    
    return '<title>数据信息预览</title>' + table + im1 + im2 + im3


def preview_file(filename, data, value_round=3):
    '''
    生成数据预览报告的html文件
    
    参数
    ----
    filename：字符串类型，文件名
    data：二维数组，数据
    value_round：整型，数字特征保留的小数点后的位数
    
    
    Generate preview report with html file
    
    Parameters
    ----------
    filename: str, file name
    data: 2-D array, data
    value_round: int, the number of digits after the decimal point retained by numeric features
    '''
    root = _preview_process(data=data, value_round=value_round)
    html = etree.HTML(root)
    tree = etree.ElementTree(html)
    tree.write(filename)


def preview(data, value_round=3):
    '''
    在jupyter中显示数据预览报告
    
    参数
    ----
    data：二维数组，数据
    value_round：整型，数字特征保留的小数点后的位数
    
    
    Display preview report in jupyter
    
    Parameters
    ----------
    data: 2-D array, data
    value_round: int, the number of digits after the decimal point retained by numeric features
    '''
    root = _preview_process(data=data, value_round=value_round)
    
    from IPython.core.display import display, HTML
    display(HTML(root))


def length_pad(seq, maxlen=None, value=0, padding='pre', dtype='int32'):
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


def sample_pad(data, index=0, padding=None):
    '''
    对二维数据进行样本填充
    先对data中的每个二维数据进行遍历，以各个index列的值作为全集，再对data的每个二维数据进行填充
    如：data1 = [[0, 1],
                 [1, 2],
                 [2, 3]]
        data2 = [[2, 3],
                 [3, 4],
                 [4, 5]]
        data = (data1, data2)
        则得到输出：
        output = [array([[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, nan],
                        [4, nan]]),
                  
                  array([[0, nan],
                         [1,nan],
                         [2, 3],
                         [3, 4],
                         [4, 5]])]
    
    data：元组或列表类型，数据
    index：整型，作为扩充全集的标准列的索引
    padding：填充值，可选，默认为None
    
    
    Sample filling for 2D data
    Values of each index column will be taken as the complete set, then each two-dimensional data of data is padded
    e.g. data1 = [[0, 1],
                 [1, 2],
                 [2, 3]]
         data2 = [[2, 3],
                  [3, 4],
                  [4, 5]]
         data = (data1, data2)
         output = [array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, nan],
                          [4, nan]]),
                    
                    array([[0, nan],
                           [1,nan],
                           [2, 3],
                           [3, 4],
                           [4, 5]])]
    
    data: tuple or list, data
    index: int, the index of a standard column as an extended complete set
    padding: padding value, optional, default=None
    '''
    time_set = set()
    result = []
    if not padding:
        padding = [np.nan] * (len(data[0][0]) - 1)
    else:
        padding = list(padding)
    
    for i in range(len(data)):
        data_part = np.array(data[i], dtype=np.object)
        result.append(data_part)
        time_set = time_set | set(data_part[:, index])
    
    for i in range(len(result)):
        different_set_list = np.array([list(time_set - set(result[i][:, index]))], dtype=np.object).T
        num = len(different_set_list)
        padding_new = np.array(padding * num, dtype=np.object).reshape(num, -1)
        different_set_list = np.hstack((different_set_list, padding_new))
        result[i] = np.vstack((result[i], different_set_list))
    
    return result


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


def dataloader(data, targets, choose_rate=0.3, is_shuffle=True, random_state=None):
    '''
    数据随机生成器
    
    参数
    ----
    data：数据
    targets：指标
    choose_rate：浮点数类型，可选，生成率，范围为[0, 1]，默认为0.3
    is_shuffle：布尔类型，可选，True表示打乱数据，False表示不打乱数据，默认为True
    random_state：整型，可选，随机种子
    
    返回
    ----
    生成器
    
    
    Data Random Generator
    
    Parameters
    ----------
    data: data
    targets: targets
    choose_rate: float, callable, generation rate whose range is [0, 1], default=0.3
    is_shuffle: bool, callable, 'True' will shuffle the data, 'False' will not, default = True
    random_state: int, callable, random seed
    
    Return
    ------
    generator
    '''
    data = np.array(data)
    targets = np.array(targets)
    
    if is_shuffle:
        np.random.seed(random_state)
        data, targets = shuffle(data, targets)
    num = len(data)
    choose_rate = int(num * choose_rate)
    times = int(math.ceil(num / choose_rate))
    
    for i in range(times):
        loc_1 = i * choose_rate
        loc_2 = (i + 1) * choose_rate
        yield data[loc_1: loc_2], targets[loc_1: loc_2]


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
        else:
            self.rc_mat = _rc(*arg)
    
    
    def __img_process(self, index, cmap):
        plt.matshow(self.rc_mat, cmap=cmap)
        plt.colorbar()
        if index:
            n_list = range(len(index))
            plt.xticks(n_list, index)
            plt.yticks(n_list, index)
    
    
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
        self.__img_process(index=index, cmap=cmap)
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
        self.__img_process(index=index, cmap=cmap)
        plt.savefig(filename)