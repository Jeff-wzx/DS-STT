# sys
import os
import glob
import numpy as np
import pickle
# torch
import torch

from data_processor import GenerateData
# 输入数据处理，处理数据集，torch.utils.data.Dataset数据迭代器，数据读取机制
# data_path是数据集的数据路径
# data_cache是数据缓存（看了一下就是存放的一些参数---在train.yaml和test.yaml里面），存储预处理数据的缓存文件路径


class Feeder(torch.utils.data.Dataset):
    # init中的在对象创建时自动调用
    def __init__(self,
                 data_path,
                 data_cache,
                 train_percent=0.8,
                 train_val_test='train'):
        # 构造函数初始化实例变量，提供的参数被分配给实例变量
        self.data_path = data_path
        self.data_cache = data_cache
        self.train_val_test = train_val_test
        # 加载数据
        self.load_data()

        # 统计数据集数量大小，看下有几个元素
        total_num = len(self.all_data)
        # equally choose validation set，
        # 从训练数据集中拆分出训练和验证数据集
        # linspace划分间隔，训练集和验证集索引是根据参数确定的。 用于将索引从0平均划分为间隔数。然后将这些间隔转换为整数并存储在变量中
        # 训练数据集的数量为0.8*数据集总数,验证集唯一元素的无序集合
        train_id_list = list(np.linspace(0, total_num-1, int(total_num*train_percent)).astype(int))
        val_id_list = list(set(list(range(total_num))) - set(train_id_list))
        # 对训练数据集进行拆分
        # last 20% data as validation set
        if train_val_test.lower() == 'train':
            self.all_data = self.all_data[train_id_list]
        elif train_val_test.lower() == 'val':
            self.all_data = self.all_data[val_id_list]

    # 加载数据模块，加载完数据后，将数据加载进入缓存文件中，如果文件已经存在，直接从缓存文件中加载数据。
    def load_data(self):
        # 为训练数据准备文件路径，调用函数以生成训练数据，并在数据生成过程中提供状态消息
        # 如果缓存文件不存在，则使用该函数生成数据并将其保存在缓存文件中(即若数据之前没有经过预处理和缓存，则执行以下步骤生成并对数据进行预处理)
        if not (os.path.exists(self.data_cache)):
            # glob.glob() 查找符合特定规则的文件路径名，返回所有匹配的文件路径列表。文件路径匹配的函数，用于获取指定的文件路径列表
            # glob获取指定目录中所有以.txt为扩展名的文件的文件路径
            # 为训练数据准备的文件路径，生成训练数据，并将其保存以供以后使用。
            # sorted是排序函数（升序还是降序看情况，默认升序，倒序看标志位reverse）

            train_file_path_list = sorted(
                 glob.glob(os.path.join(self.data_path, 'prediction_train/*.txt')))
            print('Generating Training Data.')
            GenerateData(train_file_path_list,self.data_path, is_train=True)

            test_file_path_list = sorted(
                glob.glob(os.path.join(self.data_path, 'prediction_test/*.txt')))
            print('Generating Testing Data.')
            GenerateData(test_file_path_list,self.data_path, is_train=False)

            with open(self.data_cache, 'rb') as reader:
                # [self.all_data,self.mean,self.std] = pickle.load(reader)
                [self.all_data] = pickle.load(reader)

        else:
            with open(self.data_cache, 'rb') as reader:
                [self.all_data] = pickle.load(reader)

    # 返回数据集中的数据样本总数，计算数据长度
    def __len__(self):
        return len(self.all_data)

    # 可以让对象实现迭代功能，获取实例对象的具体内容
    # 此方法返回给定索引（）的数据示例。idx
    # 使用索引访问数据并将其存储在变量中。data
    # 方法返回与指定索引（针对序列）或键（针对映射）相关联的值，使用 对象[index] 或者 对象[key] 将自动调用该方法。
    # 索引的话就是寻找0~（n-1）之间的一个数；映射的话就是找字典中的键。
    def __getitem__(self, idx):
        # 是一个列表，其中包含数据集的预处理数据。self.all_data的每个元素表示一个数据样本。
        # 该方法用于根据其索引检索特定数据样本。
        # 从数据集中检索给定索引（idx）处的单个数据样本。返回的数据取决于是用于训练还是测试
# copy用于创建数据示例的副本。这样做是防止对数据副本所做的任何更改影响原始数据
        # 这有助于保持原始数据集的完整性，并确保在训练或评估期间独立修改每个数据样本
        data = self.all_data[idx].copy() 
        # 在训练阶段使用数据增强
        if self.train_val_test.lower() == 'train':
            # 这里应该是在数据增强，随机旋转，th生成0到2pi之间的随机角度，
            # 旋转倒数第二个（-2）和倒数第一个（-1）特征尺寸；旋转平均矢量的第一个元素（0）和第二个元素（1）.
            # 随机旋转特征和平均值，可以在数据中引入了可变性，有助于提高模型在训练期间的泛化和鲁棒性
            th = np.random.random() * np.pi * 2
            data['features'][:, :, -2] = data['features'][:, :, -2] * np.cos(th) - data['features'][:, :, -1] * np.sin(th)
            data['features'][:, :, -1] = data['features'][:, :, -2] * np.sin(th) + data['features'][:, :, -1] * np.cos(th)
            data['mean'][0] = data['mean'][0] * np.cos(th) - data['mean'][1] * np.sin(th)
            data['mean'][1] = data['mean'][0] * np.sin(th) + data['mean'][1] * np.cos(th)
        # 判断在训练（或验证）阶段还是测试阶段，从而返回不同的值。（特征、掩码、平均值、原点值、邻居）
        if self.train_val_test.lower() == 'test':
            # return data['features'],data['masks'],data['mean'],data['origin'],self.mean,self.std
            return data['features'],data['masks'],data['mean'],data['origin'],data['neighbors']
        else:
            return data['features'],data['masks'],data['mean'],data['neighbors']
    # 处理单个样本的时候，使用到了数据增强，返回单个样本及其标签