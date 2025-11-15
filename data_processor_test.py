import numpy as np
import glob
import os
import pickle
from tqdm import tqdm
from scipy import spatial
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

data_root = './data/ApolloScape/'

history_frames = 6
future_frames = 6
total_frames = history_frames + future_frames
frame_step=1
feature_id=[3,4,2,9,6,7]
max_object_nums = 115
neighbor_distance = 15

def GenerateData(file_path_list, data_root, is_train=True):
    all_data = []

    for file_path_idx in tqdm(file_path_list):
        print(file_path_idx)
        with open(file_path_idx, 'r') as reader:
            content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)

        scene_frames = content[:, 0].astype(np.int64)        
        unique_frames = sorted(np.unique(scene_frames).tolist())
        if is_train:
            # 在debug中 unique_frames的值在第一个训练数据集的txt中为（0到36）
            # start_frame_ids=（0到25，第零帧到-total_frames+1）,从unique_frames中从第0个元素开始提取到倒数第11个元素
            start_frame_ids = unique_frames[:-total_frames+1]
        else:
            start_frame_ids = unique_frames[::history_frames]
        data_list = []
        for start_index in start_frame_ids:
            # 训练的时候采样帧，采12帧，比如（0，12），（1，13）...[包括起始帧不包括最后一帧]
            if is_train:
                sample_frames = np.arange(start_index, start_index + total_frames)
            else:
                # 在测试的时候，选取六帧，(0，6), (6,12), (12,18)...
                sample_frames = np.arange(start_index, start_index + history_frames)
            # 比较两个数组，返回一个布尔掩码，比如在第一次的时候让他在使用（0到11帧为true，屏蔽11帧以后的所有为false）
            # （-1，1）转换为二维列向量，（1，-1）转换为二维行向量，场景帧的内容为数据集txt文件中的第一列，采样帧为12帧（0，11帧）
            # all相当于与，any相当于或；axis=1行
            sample_mask = np.any(scene_frames.reshape(-1, 1) == sample_frames.reshape(1, -1), axis=1)
            # sample_object_ids = np.sort(np.unique(content[sample_mask, 1].astype(np.int)))
            # 选择 content 数组中满足 sample_mask 条件的行，并提取这些行中的第1（从0开始算，即第二列）列的值。
            # 在这12帧中有重复的智能体，这个的作用是计算一共有几个不同的智能体，并取出其id值
            sample_object_ids = np.unique(content[sample_mask, 1].astype(np.int))
            # print(start_index,sample_object_ids)
            # le=len(sample_object_ids)
            # max_object.append(le)
            # 取出xy坐标，从 `content` 数组中选择满足 `sample_mask` 条件的行，并提取这些行中索引为 3 到 4 的列。并求取平均值
            # 这里是取得12帧中所有车俩行人和自行车的所有坐标。axis=0表示按列进行求平均值。
            xy_coordinate = content[sample_mask, 3:5].astype(float)
            mean_xy = np.mean(xy_coordinate, axis=0)
            # print('mean_xy',mean_xy)
            # 12帧，每帧的最大数量为115，（特征有六个，加了两个速度 特征总共=8）
            # 初始化，对象之间的邻居关系、对象的特征信息、对象的存在状态（这个存在状态应该是指其是否在这一帧内）。
            # （在测试的时候还有一个原始信息），这些数组将在后续的数据生成过程中被填充和更新，用于存储生成的数据的相关信息。
            # neighbor_mask全为false大小为（12*115*115），sample_object_input全为0大小为（12*115*8），sample_object_mask大小为（12*115）
            if is_train:
                # 注意这个neighbor_mask,在这个地方创建邻接矩阵
                centrality_matrix = np.zeros((total_frames, max_object_nums, 2), dtype=np.float32)
                neighbor_matrix_weighted = np.zeros((total_frames, max_object_nums, max_object_nums), dtype=np.float32)
                neighbor_matrix_unweighted = np.zeros((total_frames, max_object_nums, max_object_nums), dtype=np.int)
                neighbor_mask = np.zeros((total_frames, max_object_nums, max_object_nums), dtype=np.bool)
                sample_object_input = np.zeros((total_frames, max_object_nums, len(feature_id)+2), dtype=np.float32)
                sample_object_mask = np.zeros((total_frames, max_object_nums), dtype=np.bool)
            else:
                centrality_matrix = np.zeros((total_frames, max_object_nums, 2), dtype=np.float32)
                neighbor_matrix_weighted = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                neighbor_matrix_unweighted = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.int)
                neighbor_mask = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.bool)
                sample_object_input = np.zeros((history_frames, max_object_nums, len(feature_id)+2), dtype=np.float32)
                sample_object_mask = np.zeros((history_frames, max_object_nums), dtype=np.bool)
                sample_object_origin = np.zeros((history_frames, max_object_nums, 3), dtype=np.int)

            # for every frame，
            # frame和object_id指向sample_frames和sample_object_ids中的每一个
            # idx相当于是一个计数器，计数用的（索引index）
            # 遍历采样帧中的每个帧以及每个智能体，并计算邻居关系。
            # sample_object_ids为1，2，7，8，9，10，12，13，14，18，19，30，50
            # enumerate函数前面就是有俩个变量，一个是指示数1，2，3，4，5，
            # 另一个是sample_frames中的那些数即1，2，7，8，9，10，12，13，14，18，，，，
            for frame_idx, frame in enumerate(sample_frames):
                # （exist_object_idx为索引值0，1，2，3，4，5，6...11）
                exist_object_idx = []
                for object_idx, object_id in enumerate(sample_object_ids):
                    # frame and object
                    # 通过与当前帧和智能体ID匹配的条件，从content中选择匹配的对象。
                    # 找出第一列等于frame的，and(并且。与)第二列等于object_id的
                    # matched_obj 为（1*10）
                    # 如果没有匹配的对象，就跳过当前循环
                    # 每一行，一行一行遍历
                    ######################
                    ########################
                    # 这里开始处理之后的xy坐标在数据集中的位置开始变换了
                    matched_obj = content[np.logical_and(content[:, 0] == frame, content[:, 1] == object_id)]
                    if 0 == len(matched_obj):
                        continue
                    obj_feature = matched_obj[0, feature_id]
                    # 前两个即xy坐标减去xy平均值； obj_feature为（1*6）；sample_object_input为（12*115*8）
                    # 正常为[:,:,:]，这里变为了[0，0，：-2]表示第0个矩阵第零行，第零行到倒数第二列（不包括倒数第二列）
                    ################################################注意注意注意减去的平均值后续记得关注一下
                    obj_feature[:2] = obj_feature[:2]-mean_xy
                    sample_object_input[frame_idx, object_idx, :-2] = obj_feature

                    # 将相应的目标智能体设置为true状态（12*115），存在的话对应位置设置为true
                    sample_object_mask[frame_idx, object_idx] = True
                    # 把处理好的信息idx索引值填入 exist_object_idx中
                    exist_object_idx.append(object_idx)
                    # 在训练的时候sample_object_origin为数据集里的前三列（帧id,目标id，目标类型）
                    if not is_train:
                        sample_object_origin[frame_idx, object_idx,:3]=matched_obj[0, :3]
                        # print(frame_idx,object_idx,matched_obj[0, :3])
                
                # print(len(exist_object_idx)).
                # 这里跳出了上个循环 exist_object_idx 包含一个帧之内的所有目标索引值（比如第一个训练数据集txt中有28个智能体）

                for obj_id_i in exist_object_idx:
                    # sample_object_input前两列为xy坐标
                    xy_1 = sample_object_input[frame_idx, obj_id_i, :2]
                    for obj_id_j in exist_object_idx:
                        xy_2 = sample_object_input[frame_idx, obj_id_j, :2]
                        relative_cord = xy_1 - xy_2
                        flag_xy =( (abs(relative_cord[0]) > neighbor_distance) | (abs(relative_cord[1]) > neighbor_distance))
                        if flag_xy:
                            dis_xy_start = np.sqrt(relative_cord[0] ** 2 + relative_cord[1] ** 2)
                            dis_xy = np.reciprocal(np.float64(dis_xy_start))
                        else:
                            dis_xy = 0
                        # （12*115*115）
                        # 如果两个对象之间的水平坐标差（x 轴方向）或者垂直坐标差（y 轴方向）的绝对值大于 neighbor_distance，则认为它们是邻居关系
                        # 这里的 > 确实有问题哦，应该是 <
                        neighbor_matrix_unweighted[frame_idx, obj_id_i, obj_id_j] = (
                            abs(relative_cord[0]) > neighbor_distance) | (
                                abs(relative_cord[1]) > neighbor_distance)

                        neighbor_mask[frame_idx, obj_id_i, obj_id_j] = (
                            abs(relative_cord[0]) > neighbor_distance) | (
                                abs(relative_cord[1]) > neighbor_distance)
                        neighbor_matrix_weighted[frame_idx, obj_id_i, obj_id_j] = ( dis_xy )
                ## 一帧完成之后计算中心性
                ####################################################################
            for frame in range(len(sample_frames)):
                 G = nx.Graph()
                 G.add_nodes_from(range(max_object_nums))
                 adjacency_matrix = neighbor_matrix_unweighted[frame]
                 for i in range(115):
                    for j in range(115):
                         if adjacency_matrix[i][j] == 1:
                             G.add_edge(i, j)

                 degree_centrality = nx.degree_centrality(G)
                 closeness_centrality = nx.closeness_centrality(G)
                 for node in range(115):
                   centrality_matrix[frame][node][0] = degree_centrality[node]
                   centrality_matrix[frame][node][1] = closeness_centrality[node]
                #####################################################################

            # add speed x ,y in dim 4,5,计算速度
            # sample_object_input为（12*115*8）之前特征使用了6列，这是应该是使用剩下的两列。
            # new_mask(11*115*2)
            # 确定哪些位置的对象在相邻两帧之间都存在,两者都不为0的时候为true转为浮点数则为1.000
            new_mask = (sample_object_input[1:, :, :2] != 0) * (sample_object_input[:-1, :, :2] != 0).astype(float)
            # 在sample_object_input的后两个特征维度添加速度信息，
            # 这里的速度信息倾向于detax和detay；因为1秒两帧，两帧相减就是1s，即速度。（后一帧减前一帧）
            sample_object_input[1:, :, -2:] = (
                sample_object_input[1:, :, :2] - sample_object_input[:-1, :, :2]).astype(float) * new_mask
            # 将第一帧的速度信息置为0；
            sample_object_input[0, :, -2:] = 0.            

            # 在 sample_object_mask 数组的最后一个维度上添加一个新的维度。用布尔值的方式表示目标时候还在（数据集里没有了，表示消失了）
            sample_object_mask = np.expand_dims(sample_object_mask, axis=-1)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 驾驶风格后续应该是需要使用这一部分的内容，后续注意

            # refine the future masks
            # data['masks'].sum(axis=0) == history_frames表示如果过去帧都在
            # 表示在过去帧都存在的情况下对未来的掩码
            # 数据字典 data，并对其中的 masks 进行了一些处理。
            # 在训练时，data字典（键值，红色是键，后面是值）内包括特征、存在状态、xy均值、邻居状态

            # 表示在过去帧都存在的情况下对未来的掩码
            if  is_train:
                data = dict(
                    features=sample_object_input, masks=sample_object_mask, mean=mean_xy,
                    neighbors=neighbor_mask, matrix=neighbor_matrix_weighted, matrix_un=neighbor_matrix_unweighted)
                # data['masks'][history_frames-1:] = np.repeat(
                #     np.expand_dims(data['masks'][:history_frames].sum(axis=0) == history_frames, axis=0),
                #     history_frames+1, axis=0) & data['masks'][history_frames-1]
                # 通过 &（与）操作，将未来的掩码限制在过去帧都存在的情况，只保留那些在过去帧都存在的对象的掩码值为true
                # 这在干什么？
                data['masks'] = data['masks'] & data['masks'][history_frames-1]
            else:
                data = dict(
                    features=sample_object_input, masks=sample_object_mask, mean=mean_xy, 
                    origin=sample_object_origin, neighbors=neighbor_mask, matrix=neighbor_matrix_weighted, matrix_un=neighbor_matrix_unweighted)
                # data['masks'][history_frames-1] = np.expand_dims(
                # data['masks'][:history_frames].sum(axis=0) == history_frames, axis=0) & data['masks'][history_frames-1]
                data['masks'] = data['masks'] & data['masks'][history_frames-1]

            data_list.append(data)
        
        all_data.extend(data_list)

    all_data = np.array(all_data)  # Train 5010 Test 415
    print(np.shape(all_data))

    # save training_data and training_adjacency into a file.
    if is_train:
        save_path=os.path.join(data_root, 'train_data.pkl')
    else:
        save_path=os.path.join(data_root, 'test_data.pkl')
    # 全部写入
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data], writer)


if __name__ == '__main__':
    train_file_path_list = sorted(
        glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    test_file_path_list = sorted(
        glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    print('Generating Training Data.')
    GenerateData(train_file_path_list,data_root, is_train=True)

    print('Generating Testing Data.')
    GenerateData(test_file_path_list,data_root, is_train=False)
