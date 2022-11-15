# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 279
        self.depth = 0.33
        self.width = 0.50

        # # ---------- transform config ------------ #
        # self.mosaic_prob = 1.0
        # self.mixup_prob = 1.0
        # self.hsv_prob = 1.0
        # self.flip_prob = 0.5

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4 # 如果训练过程占用内存太多，可以减少该值
        self.input_size = (640, 640)  # （高、宽）
        self.multiscale_range = 5 # 实际多尺度训练中图片范围为[640 - 5 * 32, 640 + 5 * 32]，如果需要取消则设定为0
        # self.random_size = (14, 26)  # 取消注释这行来具体设定尺度范围
        self.data_dir = None # 数据集图片的位置，如果为None则使用“dataset”文件的路径
        self.train_ann = "instances_train2017.json" # name of annotation file for training
        self.val_ann = "instances_val2017.json" # name of annotation file for evaluation
        self.test_ann = "instances_test2017.json" # name of annotation file for testing
   
        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0 # 应用 mosaic aug 的概率
        self.mixup_prob = 1.0 # 应用 mixup aug 的概率
        self.hsv_prob = 1.0 # 应用 hsv aug 的概率
        self.flip_prob = 0.5 # 应用 flip aug 的概率
        self.degrees = 10.0 # 旋转角度区间 这里是 (-10，10）
        self.translate = 0.1 # 转换区间 这里是 (-0.1, 0.1)
        self.mosaic_scale = (0.1, 2)  # 马赛克尺度
        self.enable_mixup = True # 是否应用 mixup aug
        self.mixup_scale = (0.5, 1.5)  # mixup aug 尺度
        self.shear = 2.0 # shear angle 区间，这里是 (-2, 2)
 
        # --------------  training config --------------------- #
  
        self.warmup_epochs = 5 # warmup epoch 数
        self.max_epoch = 300 # 最大训练 epoch
        self.warmup_lr = 0 # warmup 期间最小的 learning rate
        self.min_lr_ratio = 0.05 # 最小的 learning rate
        self.basic_lr_per_img = 0.01 / 64.0 # learning rate for one image. During training, lr will multiply batchsize.
        self.scheduler = "yoloxwarmcos" # LRScheduler 名字
        self.no_aug_epochs = 15 # 最后 n 轮不使用 augmention like mosaic
        self.ema = True # 在训练中采用 EMA 
 
        self.weight_decay = 5e-4 # 优化器的 weight_decay
        self.momentum = 0.9 # 优化器的 momentum
        self.print_interval = 5 # 迭代中输出日志的频率，如果设定为1则每次都输出log
        self.eval_interval = 30 # 训练过程中评估的 epoch 频率，这里每10个 epoch 会评估一次
        self.save_history_ckpt = True # 是否保留训练历史的 checkpoint，如果是 False 则只有 latest and best ckpt
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]  # 实验名字
 
        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)  # evaluation/test 期间的图片大小
        self.test_conf = 0.01 # evaluation/test 的输出阈值，可能性大于该阈值的预测结果才会被输出
        self.nmsthre = 0.65 # nms 阈值

        

        # self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(
                data_dir=os.path.join(get_yolox_datadir(),"VOCHaiShan"),
                image_sets=[('2007', 'trainval')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import VOCDetection, ValTransform

        valdataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCHaiShan"),
            image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
