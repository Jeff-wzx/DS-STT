import time
import adamod
import yaml
import zipfile
from tqdm import tqdm
from arg_parser import get_parser
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from model import s2tnet
from feeder import Feeder
from utilies import *

# 获取当前本地时间并将其格式化为字符串，冒号：代替为_，符号替换

# 原来localtime = time.asctime(time.localtime(time.time()))时间格式有问题，文件格式名问题（可能是版本问题）替换成上面没错了
# 此行从类创建一个对象。它用于将事件和数据写入TensorBoard。该参数指定将保存事件的目录。变量与字符串连接以创建目录路径。
# write是一个文件夹，文件夹下面内容以时间方式命名，里面装的就是可以被tensorboard所解释的文件。
# 为了更好的观察训练过程，记录处理过程的信息，SummaryWriter用来记录训练过程中的学习率和损失函数的变化。
##########################
localtime = time.asctime(time.localtime(time.time()))
x_writer = SummaryWriter('writer/', localtime)
# 定义了将预测的结果的文件名和路径进行保存
test_result_file = 'prediction_result.txt'
# 检测GPU是否可以运行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Processor():
    """
        Processor for s2tnet
        s2tnet的处理程序、网络和数据的处理部分
    """
    # init括号的参数中最起码要有一个self；相当于声明，把属性设置在这一块里，初始化各种属性并执行一些初始化和设置
    # 保存值（参数）、加载数据、加载模型、加载优化器、初始化最佳位移误差（最终和平均）；
    # 将字符串写入打开的文件。它似乎通过将“start”消息写入日志文件来启动新日志。用了with open 之后就不需要在使用close关闭文件了。
    # 初始化参数
    # work_dir工作路径为根目录下的checkpoints下的文件（数据集 Apollo）里
    # 判断训练还是测试阶段，在训练前加载检查点。（打开的文件夹）
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.best_fde=5
        self.best_ade=1.25
        with open('{}/log.txt'.format(self.arg.work_dir), 'w') as f:
            print('start', file=f)

    # 首先判断一下是测试还是训练阶段，控制模型训练和测试的方法
    def start(self):
        if self.arg.phase == 'train':
            if self.arg.load_checkpt:
                # 如果为真，加载检查点中的模型和默认ade值
                self.load_checkpoint(self.arg.test_model,self.arg.ade)
            # 迭代start_epoch=0，num_epoch=120
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # 评估模型时的一个标志，满足评估间隔（5）或是最后一个epoch时会进行模型的评估，选出误差最小的epoch
                # 但是看在config.yaml文件中eval_interval设置的是1，则说明每个epoch都会进行计算
                eval_model_flag = ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)
                ######################Training##################(这个标志位符号建议还是改改，会产生歧义)
                if self.arg.val_test:
                    self.val_epoch(epoch)
                else:
                    self.train_epoch(epoch)
                    ######################valuing##################
                    if eval_model_flag or epoch > self.arg.eval_interval:
                        wsade,wsfde=self.val_epoch(epoch)
                        # 相当于是之前默认设了一个最佳值，后续算出来的值与之进行比较，最终选出最佳值。
                        if wsade < self.best_ade:
                            self.best_ade = wsade 
                            self.best_epoch = epoch 
                            self.best_fde = wsfde
                            self.save_checkpoint(self.best_epoch,self.best_ade)

        if self.arg.phase == 'test':
            self.load_checkpoint(self.arg.test_model,self.arg.ade)
            self.test_epoch()

    # 训练epoch阶段 # 加载模型，model.train()：启用Batch Normalization和Dropout
    def train_epoch(self, epoch):
        # 将模型设置为训练模式并输出当前epoch。model.train()主要是对模型涉及到的归一化，dropout
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch))
        # 加载数据
        loader = self.data_loader['train']
        lr = self.arg.base_lr
        if self.arg.optimizer != 'NoamOpt': # is not，如果优化器不是NoamOpt，则调整学习率
            lr = self.adjust_learning_rate(epoch)
        # 创建空列表，用于存储每个批次的损失值
        loss_value = []
        self.record_time()
        # 定义了一个字典来跟踪在不同操作上花费的时间
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        # for batch_idx, (features,masks,mean_xy,neighbors) in enumerate(loader):
        # 在训练数据loader中的批处理上进入循环,以及其索引值
        for batch_idx, batch_data in enumerate(loader):
            # 批处理数据加载到特征、掩码、均值（64，2）、邻居当中
            # batch_in将特征（64，12，15，8）、掩码（64，12，115，1）、邻居（64，12，115，115）整合在一起
            features,masks,mean,neighbors = batch_data
            batch_in = features,masks,neighbors
            # 表示特征的帧数（64，12，15，8），time_horizon结果为12
            time_horizon = features.shape[1]
            timer['dataloader'] += self.split_time()
            # 从第二帧开始遍历，并调用模型预测输出轨迹了，即遍历[2,12)
            # 当前帧的掩码将从相应的设备中提取并移动到相应的设备（device为cuda0）
            for current_frame in range(2, time_horizon):
                predicted, _ = self.model(
                    batch_in,current_frame,device,is_train = True)

                mask = masks[:, current_frame:].to(device)
                ground_truth = features[:, current_frame:,:,-2:].to(device)
                predict_traj = predicted * mask
                ground_truth = ground_truth * mask
                # backward 梯度归零
                self.optimizer.zero_grad()
                # 误差为真实值和预测值之间的绝对差值
                error_order=1
                error=torch.abs(predict_traj - ground_truth) ** error_order
                # 沿着第三维度和第二维度压缩，求和
                error = error.sum(dim=3).sum(dim=2)
                overall_mask = mask.sum(dim=3).sum(dim=2)
                # 损失的计算方式为误差总和除以总掩码和 1 之间的最大值，以避免除以零
                loss = error.sum() / torch.max(overall_mask.sum(), torch.ones(1,).to(device))
                # 反向传播
                loss.backward()

                # total_loss.backward()
                if self.arg.optimizer == 'NoamOpt':
                    self.optim.step()
                else:
                    self.optimizer.step()

                timer['model'] += self.split_time()
                loss_value.append(loss.data.item())

                # record log，记录日志，记录使用的epoch,迭代、损失值、学习率，
                if batch_idx % self.arg.log_interval == 0:
                    # x_writer.add_graph(self.model,input_to_model=(input_data,origin_A,False))
                    step = epoch * len(loader) + batch_idx
                    ##############################################
                    x_writer.add_scalar('loss-Train', loss.data.item(), step)
                    if self.arg.optimizer == 'NoamOpt':
                        self.print_log(
                        '\t|Epoch:{:>5}/{:>5}|\tIteration:{:>5}/{:>5}|\tLoss:{:.5f}|lr: {:.4f}|'.format(
                        epoch, self.arg.num_epoch,batch_idx, len(loader), loss.data.item(), self.optim._rate))
                    else:
                        self.print_log(
                        '\t|Epoch:{:>5}/{:>5}|\tIteration:{:>5}/{:>5}|\tLoss:{:.5f}|lr: {:.4f}|'.format(
                        epoch, self.arg.num_epoch,batch_idx, len(loader), loss.data.item(), lr))  


    # 使用装饰器，使得在本函数运行的代码都不产生梯度信息。
    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch))

        loader = self.data_loader['val']
        # 存储累积误差、元素计数。
        cat_mask =[]
        sum_list = []
        number_list = []
        h_len = self.arg.history_len
        for batch_data in loader:
            features,masks,mean,neighbors = batch_data
            batch_in = features,masks,neighbors
            b,t,v,c = features.shape
            decoder_input = torch.zeros((b, 1, v, 2)).to(device) 
            for i in range(h_len):
                
                predicted, att = self.model(
                    batch_in, h_len, device, decoder_input, False)
                # 这行代码的目的是将每个样本的下一个时间步的预测值添加到 decoder_input，
                # 以便在下一个时间步中使用这些预测值。这是典型的递归或迭代预测方法，
                # 其中模型不断生成下一个时间步的预测，然后将其用作输入以生成下一个时间步的预测
                # 最后的1是来连接的轴，表示时间轴
                decoder_input = torch.cat((decoder_input, predicted[:, -1:]), 1)
            # 对第一个时间步后的所有时间进行【cumsum(1)】求和
            predicted_xy=decoder_input[:, 1:].cumsum(1)
            predicted_trajectory = predicted_xy + features[:,h_len-1:h_len, :, :2].to(device)
            ground_truth = features[:, h_len:,:,:2].to(device) * masks[:, h_len:].to(device)
            predict_traj = predicted_trajectory * masks[:, h_len:].to(device)

            error_order=2
            error=torch.abs(predict_traj - ground_truth) ** error_order
            error = error.sum(dim=3).sum(dim=2)
            overall_mask = masks[:, h_len:].sum(dim=3).sum(dim=2)

            number_list.extend(overall_mask.detach().cpu().numpy())
            sum_list.extend(error.detach().cpu().numpy())

        sum_time = np.sum(np.array(sum_list)**0.5, axis=0)
        num_time = np.sum(np.array(number_list), axis=0)
        overall_loss_time = (sum_time / num_time)
        overall_log = '[{:>15}] [FDE: {:.3f}] [ADE: {:.3f}] [best_FDE: {:.3f}] [best_ADE: {:.3f}] 7--12s: {}'.format(
            'Unweighted Sum', overall_loss_time[-1], np.mean(overall_loss_time),self.best_fde,self.best_ade,
            ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))

        self.print_log(overall_log)

        WSADE=np.mean(overall_loss_time)
        WSFDE=overall_loss_time[-1]
        
        info = {
                'Ade': WSADE,
                'Fde': WSFDE
                }
        ##############################
        for tag, value in info.items():
            x_writer.add_scalar(tag, value, epoch)

        return WSADE,WSFDE

    def test_epoch(self):
        self.model.eval()

        with open(test_result_file, 'w') as writer:
            loader = self.data_loader['test']
            h_len = self.arg.history_len
            for batch_data in tqdm(loader):                
                features,masks,mean,origin,neighbors = batch_data
                batch_in = features,masks,neighbors
                b,t,v,c = features.shape
                decoder_input = torch.zeros((b, 1, v, 2)).to(device) 
                for i in range(h_len):

                    predicted,att = self.model(
                        batch_in, h_len, device, decoder_input, False)
                    
                    decoder_input = torch.cat((decoder_input, predicted[:, -1:]), 1)

                predicted_xy=decoder_input[:, 1:].cumsum(1) 
                predicted_trajectory = predicted_xy + features[:,h_len-1:h_len, :, :2].to(device)

                now_pred = predicted_trajectory.detach().cpu().numpy()
                now_mean_xy = mean.detach().cpu().numpy()
                now_mask = masks[:, -1].detach().cpu().numpy()
                origin=origin.detach().cpu().numpy()
                # batch
                for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, origin, now_mask):                
                    # time
                    for time_ind, n_pre in enumerate(n_pred):
                        #nodes 
                        for info, pred, mask in zip(n_data[-1], n_pre+n_mean_xy, n_mask):
                            if mask:
                                information = info.copy()
                                information[0] = information[0] + time_ind + 1
                                result = ' '.join(information.astype(str)) \
                                        + ' ' + ' '.join(pred.astype(str)) + '\n'
                                writer.write(result)

        with zipfile.ZipFile('prediction_result.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(test_result_file)

    def save_checkpoint(self, epoch,ade):
        filename='epoch_{:04}_{:06.00f}.pt'.format(epoch,ade*10000)
        try:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                        }, os.path.join(self.arg.work_dir, filename))

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)

# {：04}宽度为4位的整数值。如有必要，它会在前面零填充整数，凑齐四位整数。
# {:06.00f}是一个占位符，表示宽度为 6 位的浮点值，包括小数点和小数部分。
# 文件名字的格式是best的epoch和平均位移误差，
    # 加载模型的检查点，从保存的模型状态、优化器状态和相关信息加载检查点
    # 文件名称，确定加载路径，从路径中加载检查点（其中包含模型状态、优化器状态和相关信息加载检查点）
    # 使得程序能够从上次中断的地方开始训练运行，而不会从头开始重新运行。
    def load_checkpoint(self, best_epoch, ade):
        filename = 'epoch_{:04}_{:06.00f}.pt'.format(best_epoch, ade)
        # filename = 'epoch_{:04}_{:04.01f}.pt'.format(best_epoch, ade)
        #  以创建检查点文件的完整路径
        ckpt_path = os.path.join(self.arg.work_dir, filename)
        #  加载检查点文件。它读取文件的内容
        checkpoint = torch.load(ckpt_path)
        # 它使用加载的检查点的纪元值更新“self.arg.start_epoch”属性，该纪元值通常是上次停止训练的纪元。
        # 它将模型的状态字典从检查点加载到“self.model”对象中。此步骤有效地将模型的参数还原到保存检查点时的状态。
        # 它将优化器的状态字典从检查点加载到“self.optimizer”对象中。此步骤将恢复优化器的状态，包括学习速率和动量等内容。
        # 最后，它会打印一条消息，指示检查点已成功从指定路径加载。
        self.arg.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Successful loaded from {}'.format(ckpt_path))

    # 训练和验证用的是一批数据
    # feeder只是输入进所有的训练轨迹数据
    # 后面的torch.utils.data.DataLoader是将数据进行分批次，一批次里面有batch_size个数据
    def load_data(self):
        self.data_loader = dict()
        self.trainLoader = Feeder(
            self.arg.train_data_path,self.arg.train_data_cache,self.arg.train_percent,'train')
        self.testLoader = Feeder(
            self.arg.test_data_path,self.arg.test_data_cache,self.arg.train_percent,'test')
        self.valLoader = Feeder(
            self.arg.train_data_path,self.arg.train_data_cache,self.arg.train_percent,'val')
        # shuffle打乱，洗牌，一般打乱比较好，num_workers进行多线程来读数据（网上看到的一般设置为0）
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.trainLoader,
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker)
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=self.valLoader,
            batch_size=self.arg.val_batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.testLoader,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker)

    def load_model(self):

        self.model = s2tnet(d_model=self.arg.d_model)
        # 数据并行在pytorch中就是DP，就是nn.DataParallel
        # 并行计算，相当于开了多个进程，每个进程自己独立运行，然后再整合在一起。，单or多GPU运行
        self.model = nn.DataParallel(self.model)
        # 将模型加载到指定设备（此处是GPU） ; to(device)就是把数据从内存放到GPU显存
        self.model.to(device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adamod':
            self.optimizer = adamod.AdaMod(
                self.model.parameters(), lr=self.arg.base_lr, beta3=0.999)
        elif self.arg.optimizer == 'NoamOpt':
            self.optim = NoamOpt(self.arg.d_model, self.arg.factor, self.arg.warmup, 
                torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            self.optimizer=self.optim.optimizer
        else:
            raise ValueError()
            # raise函数使得我们在程序中手动设置异常，其中ValueError表示引发执行类型的异常（值异常错误）

    # save all arg in work directory with yaml format，将参数保存到yaml文件中去
    def save_arg(self):
        # save arg，vars返回对象object的属性和属性值的字典对象，获取对象的属性和属性值的字典形式，
        # 键值，键是属性名称，值是各自的值。makedirs创建目录
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            # yaml.dump就是将yaml文件一次性全部写入你创建的文件。
            yaml.dump(arg_dict, f)


    # 根据当前epoch阶段调整优化器在训练过程中的学习率；学习率提高到计划中已完成步骤数的幂次
    # 输入是epoch，初始化lr = self.arg.base_lr
    # 学习率：控制模型在每次参数更新时的步长大小。它决定了模型在每一轮迭代中对参数进行调整的程度。
    # 较小的学习率可以使模型收敛得更精确，但需要更多的训练时间。（梯度下降）
    # 较大的学习率可以加快训练速度，但导致模型无法收敛或在最优解附近震荡。
    def adjust_learning_rate(self, epoch):
        lr = self.arg.base_lr
        # arg.step中设置是什么时候（在哪一步）对学习率lr进行调整
        step = self.arg.step
        lr = self.arg.base_lr * (
            self.arg.base_lr ** np.sum(epoch >= np.array(step)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # 输出时间，时间格式 星期 月 日 时分秒 年份
    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    # 运行的时候会进行展示并且加载日志并进行保存
    # print log in screen and save in log.txt
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    # save current time
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    # get time interval from last record time to now
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def display_result(self, pra_results, predict_result_lable='Train_epoch'):
        sum_list, num_list = pra_results
        sum_time = np.sum(sum_list**0.5, axis=0)
        num_time = np.sum(num_list, axis=0)
        overall_loss_time = (sum_time / num_time)
        overall_log = '[{:>15}] [FDE: {:.3f}] [ADE: {:.3f}] 7--12s: {}'.format(
            predict_result_lable, overall_loss_time[-1],
            np.mean(overall_loss_time),
            ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))

        self.print_log(overall_log)
        return overall_loss_time

seed_torch()

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    print(p.config)

    # 判断p.config是否为空none，若不是则进行下面的操作；assert断言，调试过程中捕捉程序错误。
    # 前边都是一些参数相关的一些东西，后面的processor是主要内容
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
            # 原来 default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()