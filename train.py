from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer) # 定义模型、loss、优化器等。其中loss在lib/trains/mot.py中的类MotLoss
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    # 2022.12.22 冻结除id分支之外的其他分支
    # freeze_names = ['base', 'dla_up', 'ida_up', 'hm', 'wh', 'reg']
    # for name, child in model.named_children():
    #     if name in freeze_names:
    #         for param in child.parameters():
    #             param.requires_grad = False
    # 检查是否成功冻结参数
    # for k, v in model.named_parameters():
    #     if k not in freeze_names:
    #         print(k,':',v.requires_grad)
    #         if v.requires_grad == True:
    #             print(k)

    # 2023.04.11 用熵权法动态计算权重
    det_loss = []
    reid_loss = []
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        # 2023.04.11 用熵权法动态计算权重
        det_loss.append(opt.hm_weight * log_dict_train['hm_loss'] + opt.wh_weight * log_dict_train['wh_loss'] + opt.off_weight * log_dict_train['off_loss'])
        reid_loss.append(log_dict_train['frame_loss'])
        # 最大最小值归一化?
        # 最大值肯定是第一个epoch的loss，最小值会随着训练而变化
        # 从第二个epoch结束才开始更新权重
        if epoch > 1:
            det_losses = torch.tensor(det_loss)
            reid_losses = torch.tensor(reid_loss)
            max_det = det_losses.max()
            min_det = det_losses.min()
            max_reid = reid_losses.max()
            min_reid = reid_losses.min()
            # 最大值归一化
            # max_det_norm = (max_det - det_losses) / (max_det - min_det)
            # max_reid_norm = (max_reid - reid_losses) / (max_reid - min_reid)
            # p_det = max_det_norm / sum(max_det_norm)
            # p_reid = max_reid_norm / sum(max_reid_norm)
            # 最小值归一化
            min_det_norm = (det_losses - min_det) / (max_det - min_det)
            min_reid_norm = (reid_losses - min_reid) / (max_reid - min_reid)
            p_det = min_det_norm / sum(min_det_norm)
            p_reid = min_reid_norm / sum(min_reid_norm)
            # 信息熵，不能直接用torch.log, 因为p里必然至少有一个元素是0
            E_det = entropy(p_det)
            E_reid = entropy(p_reid)
            # 更新权重
            w_det = (1 - E_det) / (2 - (E_det + E_reid))
            w_reid = (1 - E_reid) / (2 - (E_det + E_reid))
            opt.det_weight = w_det #*10
            opt.reid_weight = w_reid

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()

def entropy(Y):
    num = Y.shape[0]
    def ln(x):
        if x > 0:
            return torch.log(x)
        else:
            return 0
    s = 0
    for i in range(num):
        s += Y[i] * ln(Y[i])
    return -(1/ln(torch.tensor(num, dtype=torch.float32))) * s

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
