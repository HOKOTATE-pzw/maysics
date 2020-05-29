'''
本模块用于评估和选择模型

This module is ued for estimating and selecting models
'''

import numpy as np
from matplotlib import pyplot as plt
from maysics import data_split


class Estimate():
    '''
    参数
    ----
    data：ndarray形式，自变量
    targets：ndarray形式，因变量
    verbose：整型，可选，可以是0、1、2，默认为0
    
    
    Parameters
    ----------
    data: ndarray
    targets: ndarray
    verbose: int, callable, can be 0, 1, 2, default = 0
    '''
    def __init__(self, data, targets, verbose=0):
        self.data = data
        self.targets = targets
        self.__verbose = verbose
    
    
    def __part_of_kfold(self, num_validation_samples, num_part):
        validation_data = self.data[num_validation_samples * num_part:
                                    num_validation_samples * (num_part + 1)]
        validation_targets = self.targets[num_validation_samples * num_part:
                                          num_validation_samples * (num_part + 1)]
        
        train_data = np.concatenate([self.data[: num_validation_samples * num_part],
                                     self.data[num_validation_samples * (num_part + 1):]])
        train_targets = np.concatenate([self.targets[: num_validation_samples * num_part],
                                        self.targets[num_validation_samples * (num_part + 1):]])
        
        return train_data, train_targets, validation_data, validation_targets


    def keras_normal(self, build_model, num_epochs, batch_size, train_size=None, val_size=None, param=None, shuffle=True, random_state=None):
        ''' 
        对keras模型或与keras模型相同架构的模型进行数据分割和训练
        
        参数
        ----
        build_model：模型函数，需要返回一个model
        num_epochs：整型，迭代次数
        batch_size：整型，批量大小
        train_size：浮点数类型，可选，训练集占总数据量的比，取值范围为(0, 1]，默认为0.75
        val_size：浮点数类型，可选，测试集占总数据量的比，取值范围为[0, 1)，当train_size被定义时，该参数无效
        shuffle：布尔类型，可选，True表示打乱数据，False表示不打乱数据，默认为True
        param：字典类型，可选，当build_model函数有参数时，需输入以参数名为键，参数值为值的字典
        random_state：整型，可选，随机种子
        
        
        Spliting data and training for keras models and other similar models
        
        Parameters
        ----------
        build_model: the function to generate keras-model, which should return a model
        num_epochs: int, the number of epochs
        batch_size: int, the size of batch
        train_size: float, callable, ratio of training set to total data, value range is (0, 1], default=0.75
        val_size: float, callable, ratio of test set to total data, value range is [0, 1)
        shuffle: bool, callable, 'True' will shuffle the data, 'False' will not, default = True
        param: dict, callable, when build_model() has non-default parameters, param needs to be input a dictionary with parm_name as key and param_value as value
        random_state: int, callable, random seed
        '''
        self.__num_epochs = num_epochs
        val_loss = np.zeros(num_epochs)
        val_acc = np.zeros(num_epochs)
        loss = np.zeros(num_epochs)
        acc = np.zeros(num_epochs)
        np.random.seed(random_state)
        
        if shuffle:
            local_data = self.data
            local_targets = self.targets
            state = np.random.get_state()
            np.random.shuffle(local_data)
            np.random.set_state(state)
            np.random.shuffle(local_targets)
        
        if not (train_size or val_size):
            train_size = 0.75
        elif val_size:
            train_size = 1 - val_size
    
        if train_size <= 0 or train_size > 1:
            raise Exception("'train_size' should be in (0, 1], 'test_size' should be in [0, 1)")
        
        num_of_data = int(len(local_data) * train_size)
        train_data = local_data[:num_of_data]
        train_targets = local_targets[:num_of_data]
        validation_data = local_data[num_of_data:]
        validation_targets = local_targets[num_of_data * train_size:]
                
        if param:
            model = build_model(**param)
        else:
            model = build_model()
        
        history = model.fit(train_data, train_targets,
                            validation_data=(validation_data, validation_targets),
                            epochs=num_epochs, batch_size=batch_size, verbose=self.__verbose)
        self.history = history.history
    
    
    def keras_rekfold(self, build_model, num_epochs, batch_size, k=5, n=3, shuffle=True, param=None, random_state=None):
        '''
        对keras模型或与keras模型相同架构的模型进行重复的k折验证
        
        参数
        ----
        build_model：模型函数，需要返回一个model
        num_epochs：整型，迭代次数
        batch_size：整型，批量大小
        k：整型，可选，k折验证的折叠数，默认为5
        n：整型，可选，k折验证的重复次数，默认为3
        shuffle：布尔类型，可选，True表示打乱数据，False表示不打乱数据，默认为True
        param：字典类型，可选，当build_model函数有参数时，需输入以参数名为键，参数值为值的字典
        random_state：整型，可选，随机种子
        
        
        Repeated k-fold verification for keras models and other similar models
        
        Parameters
        ----------
        build_model: the function to generate keras-model, which should return a model
        num_epochs: int, the number of epochs
        batch_size: int, the size of batch
        k: int, callable, the number of k-fold, default = 5
        n: int, callable, the number of repetition of k-fold, default = 3
        shuffle: bool, callable, 'True' will shuffle the data, 'False' will not, default = True
        param: dict, callable, when build_model() has non-default parameters, param needs to be input a dictionary with parm_name as key and param_value as value
        random_state: int, callable, random seed
        '''
        self.__num_epochs = num_epochs
        num_validation_samples = len(self.data) // k
        val_loss = np.zeros(num_epochs)
        val_acc = np.zeros(num_epochs)
        loss = np.zeros(num_epochs)
        acc = np.zeros(num_epochs)
        np.random.seed(random_state)
        
        for i in range(n):
            if shuffle:
                local_data = self.data
                local_targets = self.targets
                state = np.random.get_state()
                np.random.shuffle(local_data)
                np.random.set_state(state)
                np.random.shuffle(local_targets)
            for j in range(k):
                train_data, train_targets,\
                validation_data, validation_targets = Estimate.__part_of_kfold(
                    self, num_validation_samples=num_validation_samples, num_part=j)
                
                if param:
                    model = build_model(**param)
                else:
                    model = build_model()
                history = model.fit(train_data, train_targets,
                                    validation_data=(validation_data, validation_targets),
                                    epochs=num_epochs, batch_size=batch_size, verbose=self.__verbose)
                if i == 0 and j == 0:
                    new_keys = list(history.history.keys())
                    new_values_of_history = np.zeros_like(list(history.history.values()))
                
                new_values_of_history += np.array(list(history.history.values()))
        
        new_values_of_history /= (n * k)
        history_dict = dict(zip(new_keys, new_values_of_history))
        self.history = history_dict
    
    
    def __img_process(self, epochs_range, metrics_list):
        epochs_range_list = np.arange(epochs_range[0], epochs_range[1]+1)
        
        if metrics_list == 'all':
            metrics_list = list(self.history.keys())
            pic_num = int(len(metrics_list) / 2)
            metrics_list = metrics_list[pic_num :]
        else:
            pic_num = len(metrics_list)
        fig = plt.figure(figsize=(6, 4 * pic_num))
        cal_num = 1
        for metric in metrics_list:
            val_range_list = self.history['val_'+metric][epochs_range[0]-1: epochs_range[1]]
            range_list = self.history[metric][epochs_range[0]-1: epochs_range[1]]
            
            ax = fig.add_subplot(pic_num, 1, cal_num)
            ax.plot(epochs_range_list, val_range_list, label='val_'+metric)
            ax.plot(epochs_range_list, range_list, label=metric)
            ax.legend()
            ax.set_title(metric)
            
            cal_num += 1
        
    
    
    def show(self, epochs_range=None, metrics_list='all'):
        '''
        作指标值-迭代次数图像并显示
        
        参数
        ----
        epochs_range：列表类型，可选，定义图像的绘制范围，默认全部画出
        metrics_list：列表类型，指定绘图的指标函数列表，默认为全部指标
        
        
        Display the metrics - epochs image
        
        Parameters
        ----------
        epochs_range: list, callable, define the range of image, default draw all the range
        metrics_list: list, used for appointing the metrics for drawing, default='all' 
        '''
        if not epochs_range:
            epochs_range = (1, self.__num_epochs)
        Estimate.__img_process(self, epochs_range, metrics_list=metrics_list)
        plt.tight_layout()
        plt.show()
    
    
    def savefig(self, filename, epochs_range=None, metrics_list='all'):
        '''
        作指标值-迭代次数图像并保存
        
        参数
        ----
        filename：字符串类型，保存的文件名
        epochs_range：列表类型，可选，定义图像的绘制范围，默认全部画出
        metrics_list：列表类型，指定绘图的指标函数列表，默认为全部指标
        
        
        Save the metrics - epochs image
        
        Parameters
        ----------
        filename: str, file name
        epochs_range: list, callable, define the range of image, default draw all the range
        metrics_list: list, used for appointing the metrics for drawing, default='all' 
        '''
        if not epochs_range:
            epochs_range = (1, self.__num_epochs)
        Estimate.__img_process(self, epochs_range, metrics_list=metrics_list)
        plt.tight_layout()
        plt.savefig(filename)



class Search():
    '''
    对keras模型或与keras模型相同架构的模型进行网格搜索
    
    参数
    ----
    data：自变量数据
    targets：目标数据
    verbos：整型，可选，可以是0、1、2，默认为0
    
    属性
    ----
    comb：最佳组合
    
    
    Grid searching for keras model or other similar models
    
    Parameters
    ----------
    data: independent variable
    targets: dependent variable
    verbose: int, callable, can be 0, 1, 2, default = 0
    
    Atrributes
    ----------
    comb: the best combination
    '''
    def __init__(self, data, targets, verbose=0):
        self.data = data
        self.targets = targets
        self.__verbose = verbose
    
    
    def __part_of_kfold(self, num_validation_samples, num_part):
        validation_data = self.data[num_validation_samples * num_part:
                                    num_validation_samples * (num_part + 1)]
        validation_targets = self.targets[num_validation_samples * num_part:
                                          num_validation_samples * (num_part + 1)]
        
        train_data = np.concatenate([self.data[: num_validation_samples * num_part],
                                     self.data[num_validation_samples * (num_part + 1):]])
        train_targets = np.concatenate([self.targets[: num_validation_samples * num_part],
                                        self.targets[num_validation_samples * (num_part + 1):]])
        
        return train_data, train_targets, validation_data, validation_targets
    
    
    def __part_of_gridsearch(self, build_model, num_epochs, batch_size, param, k, n, select_method, verbose, shuffle):
        num_validation_samples = len(self.data) // k
        
        param_keys_list = list(param.keys())
        param_values_list = np.array(list(param.values()))
        param_values_list = np.meshgrid(*param_values_list)
        num_param_values_list = len(param_values_list)
        for i in range(num_param_values_list):
            param_values_list[i] = param_values_list[i].flatten()
        param_values_list = np.array(param_values_list)
        param_values_list = param_values_list.T
        
        val_m = []
        val_m_dict = []
        val_m_loc = []
        for param_values in param_values_list:
            param_dict = dict(zip(param_keys_list, param_values))
            val_list = np.zeros(num_epochs)
            for i in range(n):
                if shuffle:
                    local_data = self.data
                    local_targets = self.targets
                    state = np.random.get_state()
                    np.random.shuffle(local_data)
                    np.random.set_state(state)
                    np.random.shuffle(local_targets)
                for j in range(k):
                    train_data, train_targets,\
                    validation_data, validation_targets = Search.__part_of_kfold(
                        self, num_validation_samples=num_validation_samples, num_part=j)
                    model = build_model(**param_dict)
                    history = model.fit(train_data, train_targets,
                                        validation_data=(validation_data, validation_targets),
                                        epochs=num_epochs, batch_size=batch_size, verbose=verbose)
                    val_list += history.history['val_'+select_method[0]]
                    
            val_list /= (n * k)  #这是网格中一种情况的结果
            if select_method[1] == 'min':
                location_ = np.argmin(val_list)
                val_m.append(val_list[location_])
                val_m_loc.append(location_)
            elif select_method[1] == 'max':
                location_ = np.argmax(val_list)
                val_m.append(val_list[location_])
                val_m_loc.append(location_)
            
            val_m_dict.append(param_dict)
        
        if select_method[1] == 'max':
            final_loc = np.argmin(val_m)
            final_dict = val_m_dict[final_loc]
            final_val_m = val_m[final_loc]
            final_epoch = val_m_loc[final_loc]
            
            return final_dict, final_val_m, final_epoch
            self.collocation = {'collocation': final_dict,
                                'loss': final_val_m,
                                'epoch': final_epoch}
        
        elif select_method[1] == 'min':
            final_loc = np.argmax(val_m)
            final_dict = val_m_dict[final_loc]
            final_val_m = val_m[final_loc]
            final_epoch = val_m_loc[final_loc]
            
            return final_dict, final_val_m, final_epoch
            self.collocation = {'collocation': final_dict,
                                'acc': final_val_m,
                                'epoch': final_epoch}
    
    
    def gridsearch(self, build_model, num_epochs, batch_size, param, k=5, n=1, select_method=('loss', 'min'), shuffle=True, random_state=None):
        '''
        参数
        ----
        buil_model：模型函数或函数列表，需要返回一个model
        num_epochs：整型，迭代次数
        batch_size：整型，批量大小
        param：字典或字典列表类型，表示build_model函数的参数，参数名为键，待选参数值列表为值
        k：整型，可选，k折验证的折叠数，默认为5
        n：整型，可选，k折验证的重复次数，默认为1
        select_method：元组类型，评估方法，形式为('指标', 'max'或'min')，默认为('loss', 'min')
        shuffle：布尔类型，可选，True表示打乱数据，False表示不打乱数据，默认为True
        random_state：整型，可选，随机种子
        
        
        Parameters
        ----------
        build_model: the function of function list to generate keras-model, which should return a model
        num_epochs: int, the number of epochs
        batch_size: int, the size of batch
        param: dict or list of dicts, param needs to be input a dictionary with parm_name as key and param_value (list) as value
        k: int, callable, the number of k-fold, default = 5
        n: int, callable, the number of repetition of k-fold, default = 1
        select_method: tuple, method of selecting the best collocation, default = ('loss', 'min')
        shuffle: bool, callable, 'True' will shuffle the data, 'False' will not, default = True
        random_state: int, callable, random seed
        '''
        np.random.seed(random_state)
        if type(build_model).__name__ == 'function':
            final_dict, final_val_m, final_epoch =\
            Search.__part_of_gridsearch(self, build_model=build_model, num_epochs=num_epochs,\
                                        batch_size=batch_size, param=param,\
                                        k=k, n=n, select_method=select_method,\
                                        verbose=self.__verbose, shuffle=shuffle)
        else:
            last_dict_list = []
            last_val_m_list = []
            last_epoch_list = []
            num_of_models = len(build_model)
            for i in num_of_models:
                last_dict, last_val_m, last_epoch =\
                Search.__part_of_gridsearch(self, build_model=build_model, num_epochs=num_epochs,\
                                            batch_size=batch_size, param=param,\
                                            k=k, n=n, select_method=select_method,\
                                            verbose=self.__verbose, shuffle=shuffle)
                last_dict_list.append(last_dict)
                last_val_m_list.append(last_val_m)
                last_epoch_list.append(last_epoch)
            
            if select_method[1] == 'min':
                last_loc = np.argmin(last_val_m_list)
                final_dict = last_dict_list[last_loc]
                final_val_m = last_val_m_list[last_loc]
                final_epoch = last_epoch_list[last_loc]
            
            elif select_method[1] == 'max':
                last_loc = np.argmax(last_val_m_list)
                final_dict = last_dict_list[last_loc]
                final_val_m = last_val_m_list[last_loc]
                final_epoch = last_epoch_list[last_loc]
        
        self.comb = {'comb': final_dict, select_method[0]: final_val_m, 'epoch': final_epoch}



class Error():
    '''
    误差分析
    
    参数
    ----
    func：函数类型，模型的预测函数
    
    属性
    ----
    abs_error：1-D ndarray数组，绝对误差列表
    rel_error：1-D ndarray数组，相对误差列表
    abs_sort：绝对误差从大到小的排序
    rel_sort：相对误差从大到小的排序
    mae：平均绝对误差
    mape：平均绝对百分比误差
    mse：平均平方误差
    rmse：均方根误差
    sae：绝对误差和
    sse：残差平方和
    
    
    Error Analysis
    
    Parameter
    ---------
    func: function, predicting function of models
    
    Atrributes
    ----------
    abs_error: 1-D ndarray, absolute error list
    rel_error: 1-D ndarray, relative error list
    abs_sort: list of absolute values of errors sorted from large to small
    rel_sort: list of relative values of errors sorted from large to small
    mae: mean absolute error
    mape: mean absolute percentage error
    mse: mean squared error
    rmse: root mean square error
    sae: sum of absolute error
    sse: sum of squared error
    '''
    def __init__(self, func):
        self.__func = func
    
    
    def fit(self, data, target):
        data = np.array(data)
        target = np.array(target)
        self.__num_data = len(target)
        
        predict_target = self.__func(data)
        self.abs_error = abs(target - predict_target)
        self.rel_error = abs((target - predict_target) / target)
    
    @property
    def abs_sort(self):
        error_index = np.arange(self.__num_data)
        self.abs_sort = sorted(list(zip(self.abs_error, error_index)), reverse=True)
    
    @property
    def rel_sort(self):
        error_index = np.arange(self.__num_data)
        self.rel_sort = sorted(list(zip(self.rel_error, error_index)), reverse=True)
    
    @property
    def sse(self):
        return sum(self.abs_error**2)
    
    @property
    def sae(self):
        return sum(self.abs_error)
    
    @property
    def mse(self):
        return sum(self.abs_error**2) / self.__num_data
    
    @property
    def mae(self):
        return sum(self.abs_error) / self.__num_data
    
    @property
    def rmse(self):
        return (sum(self.abs_error**2) / self.__num_data)**0.5
    
    @property
    def mape(self):
        return sum(self.rel_error) / self.__num_data



class Sense():
    '''
    灵敏度分析
    r = (x0, x1, x2, ..., xn)
    y = f(r)
    第i个特征在r=r0时的灵敏度：
    s(xi, r0)= (dy/dxi) * (xi/y)   (r=r0)
    
    参数
    ----
    func：函数类型，模型的预测函数
    x_index：列表类型，可选，需要求灵敏度的特征索引列表，默认为全部
    acc：浮点数类型，可选，求导的精度，默认为0.05
    
    属性
    ----
    s_mat：由特征的灵敏度值组成的矩阵
    prediction：预测值列表
    
    
    Sensitivity Analysis
    r = (x0, x1, x2, ..., xn)
    y = f(r)
    the sensitivity of the ith feature at r=r0:
    s(xi, r0)= (dy/dxi) * (xi/y)   (r=r0)
    
    Parameters
    ----------
    func: function, predicting function of models
    x_index: list, callable, index of features whose sensitivity needs to be calculated, default to all
    acc: float, callable, accuracy of derivation, default=0.05
    
    Attributes
    ----------
    s_mat: matrix combined of sensitivities of features
    prediction: list of predicted values
    '''
    def __init__(self, func, x_index=None, acc=0.05):
        self.__func = func
        self.x_index = x_index
        self.acc = acc
        self.s_mat = []
        self.prediction = []
    
    def fit(self, x0):
        '''
        参数
        ----
        x0：特征的初始值
        
        
        Parameter
        ---------
        x0: initial values of features
        '''
        if not self.x_index:
            self.x_index = range(len(x0))
        
        acc2 = 2 * self.acc
        func0 = self.__func(x0)
        self.prediction.append(func0)
        s_list = []
        
        for i in self.x_index:
            x0[i] += self.acc
            func1 = self.__func(x0)
            x0[i] -= acc2
            func2 = self.__func(x0)
            de = (func1 - func2) / (acc2 * func0) * x0[i]
            x0[i] += self.acc
            s_list.append(de)
        
        self.s_mat.append(s_list)
    
    
    def clr(self):
        '''
        清空s_mat和prediction
        
        
        clear the s_mat and the prediction
        '''
        self.s_mat = []
        self.prediction = []