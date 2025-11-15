import argparse
import sys
import os
from arg_types import arg_boolean, arg_dict

dataset_name = 'apollo'
home_dir = os.getcwd()

# os.getcwd()函数获得当前的路径。

# 该函数将布尔值的字符串表示形式转换为其相应的布尔值，变量参数v变小写
# 目的是处理布尔值作为字符串提供的情况，并将其转换为正确的布尔值
# 字符串的处理str型转换成为bool型


def str2bool(v):
    # 符合这里边的返回布尔值真true
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    # 符合这里边的返回布尔值假false
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # 指示输入不是有效的布尔值。
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 1：导入包import
# 2：获取参数，建立解析对象。
def get_parser():
    # 参数的优先级：命令行 > 配置的> 默认
    # parameter priority: command line > config > default
    # 该参数用于提供程序的简要说明：(空时图卷积网络)，程序描述。建立解析对象。
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    # 3：对象值赋参(工作路径，配置文件路径，阶段（训练、测试、验证），加载间隔（加载）)
    parser.add_argument('--work_dir', default=home_dir+'/checkpoints/' + dataset_name)
    parser.add_argument('--config', default=home_dir+'/config/apolloscape/train.yaml')
    # 上面最后的test.yaml原来是train.yaml，config文件会根据这个文件中的参数对config文件中的内容进行相应的修改
    # 训练的时候改为train.yaml；测试的时候改为test.yaml
    parser.add_argument('--phase', default='train')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='the interval for printing messages (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True,
                        help='print logging or not')
    parser.add_argument('--test_model', type=int, default=5)
    parser.add_argument('--load_checkpt', type=str2bool, default=False)

    parser.add_argument(
        '--feeder', default='feeder.Feeder', help='data loader will be used')
    # 工作器数量和训练测试数据的路径
    # 网上看到num_worker设置为0看看，不然容易报错
    parser.add_argument('--num_worker', type=int, default=10)
    parser.add_argument('--train_data_path', default=' ')
    parser.add_argument('--test_data_path', default=' ')
    # 缓存器存储路径，将数据集进行拆分，训练集中的百分之八十用于训练，剩下的百分之二十用于验证
    parser.add_argument('--train_data_cache', default=' ')
    parser.add_argument('--test_data_cache', default=' ')
    parser.add_argument('--train_percent', type=float, default=0.8)
    # 当学习率设置的过小时，收敛过程将变得十分缓慢（学习率不能过低，否则到后面的时候就会出现梯度消失的情况）。
    # 当学习率设置的过大时，梯度可能会在最小值附近来回震荡，甚至可能无法收敛。
    # 初始的学习率、以及学习率在什么时候变化、优化器、批量大小（一次输入进多少数据）(训练、验证和测试的)
    parser.add_argument(
        '--base_lr', type=float, default=0.1, help='initial learning rate')
    # nargs='+'应该读取的命令行参数个数，可以是具体数字或者符号，比如+表示1或者多个参数，*表示多个参数或者0。
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--optimizer', default='Adam',help='type of optimizer')
    parser.add_argument(
        '--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--val_batch_size', type=int, default=256, help='value batch size')
    # epoch从第几个开始以及一共有多少个
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=120,
                        help='stop training in which epoch')
    # 权值衰减目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题，所以权重衰减也叫L2正则化。
    # weight_decay是一项正则化技术，抑制模型的过拟合，从而来提高模型的泛化能力。模型权重数值越小，模型的复杂度越低。
    # 初始的ade参数，比例系数、模型的维度、预测长度、历史长度、验证还是测试阶段。
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay for optimizer')
    parser.add_argument('--ade', type=float, default=800.0)
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--predict_len', type=int, default=6)
    # 三秒，一秒两帧，2*3=6。
    parser.add_argument('--history_len', type=int, default=6)
    parser.add_argument('--val_test', type=str2bool, default=False)
    return parser


