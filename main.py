# -*- coding: utf-8 -*-
import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def ensureData(dir_img):
    filepaths = []
    if not os.path.exists(dir_img):
        print(f"警告: 目录 {dir_img} 不存在")
        return filepaths

    flv = os.listdir(dir_img)
    for f in flv:
        if f.split(".")[-1].lower() in ["jpg", "jpeg", "png", "bmp"]:
            filepaths.append(os.path.join(dir_img, f))
    return filepaths


def get_datasets(params, input_size):

    train_imgs = "../datasets/out_roco/images/"
    test_imgs = "../datasets/test_roco/images/"

    train_set = ensureData(train_imgs)
    test_set = ensureData(test_imgs)

    if len(train_set) == 0:
        raise ValueError("训练集为空，请检查数据集路径")
    if len(test_set) == 0:
        print("警告: 测试集为空，将使用训练集进行测试")
        test_set = train_set[: min(100, len(train_set))]  # 取部分训练数据作为测试

    print(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")

    # 创建数据集对象
    train_dataset = Dataset(train_set, input_size, params, augment=True)
    test_dataset = Dataset(test_set, input_size, params, augment=False)

    return train_dataset, test_dataset, train_set, test_set


def train(args, params, train_dataset, test_dataset, train_set, test_set):
    # Model
    model = nn.yolo_v11_n(len(params["names"]))
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params["weight_decay"] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(
        util.set_params(model, params["weight_decay"]),
        params["min_lr"],
        params["momentum"],
        nesterov=True,
    )

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(train_dataset)

    loader = data.DataLoader(
        train_dataset,
        args.batch_size,
        sampler is None,
        sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    with open("weights/step.csv", "w") as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(
                log,
                fieldnames=[
                    "epoch",
                    "box",
                    "cls",
                    "dfl",
                    "Recall",
                    "Precision",
                    "mAP@50",
                    "mAP",
                ],
            )
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                train_dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(("\n" + "%10s" * 5) % ("epoch", "memory", "box", "cls", "dfl"))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast("cuda"):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= (
                    args.world_size
                )  # gradient averaged between devices in DDP mode
                loss_cls *= (
                    args.world_size
                )  # gradient averaged between devices in DDP mode
                loss_dfl *= (
                    args.world_size
                )  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.4g}G"  # (GB)
                    s = ("%10s" * 2 + "%10.3g" * 3) % (
                        f"{epoch + 1}/{args.epochs}",
                        memory,
                        avg_box_loss.avg,
                        avg_cls_loss.avg,
                        avg_dfl_loss.avg,
                    )
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema, test_dataset)

                logger.writerow(
                    {
                        "epoch": str(epoch + 1).zfill(3),
                        "box": str(f"{avg_box_loss.avg:.3f}"),
                        "cls": str(f"{avg_cls_loss.avg:.3f}"),
                        "dfl": str(f"{avg_dfl_loss.avg:.3f}"),
                        "mAP": str(f"{last[0]:.3f}"),
                        "mAP@50": str(f"{last[1]:.3f}"),
                        "Recall": str(f"{last[2]:.3f}"),
                        "Precision": str(f"{last[3]:.3f}"),
                    }
                )
                log.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {"epoch": epoch + 1, "model": copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f="./weights/last.pt")
                if best == last[0]:
                    torch.save(save, f="./weights/best.pt")
                del save

    if args.local_rank == 0:
        util.strip_optimizer("./weights/best.pt")  # strip optimizers
        util.strip_optimizer("./weights/last.pt")  # strip optimizers


@torch.no_grad()
def test(args, params, model=None, test_dataset=None):
    if test_dataset is None:
        # 如果没有提供测试数据集，则创建一个
        test_imgs = "../datasets/test_roco/images/"
        test_set = ensureData(test_imgs)
        test_dataset = Dataset(test_set, args.input_size, params, augment=False)

    loader = data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    plot = False
    if not model:
        plot = True
        if os.path.exists("./weights/best.pt"):
            model = torch.load(f="./weights/best.pt", map_location="cuda")
            model = model["model"].float().fuse()
        else:
            print("警告: 未找到预训练模型，使用随机初始化模型进行测试")
            model = nn.yolo_v11_n(len(params["names"])).fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(
        start=0.5, end=0.95, steps=10
    ).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(
        loader, desc=("%10s" * 5) % ("", "precision", "recall", "mAP50", "mAP")
    )
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.0  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets["idx"] == i
            cls = targets["cls"][idx]
            box = targets["box"][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append(
                        (metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1))
                    )
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(
            *metrics, plot=plot, names=params["names"]
        )
    # Print results
    print(("%10s" + "%10.3g" * 4) % ("", m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop

    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v11_n(len(params["names"])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f"Number of parameters: {num_params}")
        print(f"Number of FLOPs: {flops}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--data-dir", default="../datasets/", type=str, help="数据集根目录"
    )

    args = parser.parse_args()

    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    args.distributed = int(os.getenv("WORLD_SIZE", 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if args.local_rank == 0:
        if not os.path.exists("weights"):
            os.makedirs("weights")

    with open("utils/args.yaml", errors="ignore") as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    # 获取数据集
    train_dataset, test_dataset, train_set, test_set = get_datasets(
        params, args.input_size
    )

    if args.train:
        train(args, params, train_dataset, test_dataset, train_set, test_set)
    if args.test:
        test(args, params, test_dataset=test_dataset)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
