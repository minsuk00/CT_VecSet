# Copyright (c) 2025, Biao Zhang.

import argparse
import datetime
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(8)

# import utils.lr_decay as lrd
import utils.misc as misc
from engines.engine_ae import train_one_epoch
from models import autoencoder
from utils.ct_dataset import CTSingleVolumeDataset
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
# from utils.objaverse import Objaverse
import wandb

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_args_parser():
    parser = argparse.ArgumentParser("VecSetAutoEncoder", add_help=False)

    parser.add_argument("--config", default="config.yaml", type=str, help="Path to config file")

    parser.add_argument("--batch_size", type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--accum_iter", type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")

    # Model parameters
    parser.add_argument("--model", type=str, metavar="MODEL", help="Name of model to train")

    parser.add_argument("--point_cloud_size", type=int, help="input size")

    # Optimizer parameters
    parser.add_argument("--clip_grad", type=float,  metavar="NORM", help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument("--weight_decay", type=float, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, metavar="LR", help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--layer_decay", type=float, help="layer-wise lr decay from ELECTRA/BEiT")

    parser.add_argument("--min_lr", type=float, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0")

    parser.add_argument("--warmup_epochs", type=int, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", type=str, help="dataset path")

    parser.add_argument("--output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--checkpoint_dir", help="path where to save heavy model checkpoints (e.g. scratch)")

    parser.add_argument("--log_dir", help="path where to tensorboard log")
    parser.add_argument("--device", help="device to use for training / testing")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", help="resume from checkpoint")

    parser.add_argument("--start_epoch", type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--dist_eval", action="store_true", help="Enabling distributed evaluation (recommended during training for faster monitor")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument("--world_size", type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", help="url used to set up distributed training")

    parser.add_argument("-W", "--no_wandb", action="store_true", help="Disable Wandb")
    
    return parser


def load_config(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update args namespace with config values
    # Command line args (like output_dir) take precedence if provided
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    return args

def main(args):
    misc.init_distributed_mode(args)
    if args.no_wandb:
        args.wandb = False

    if args.output_dir and misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        # Redirect stdout to file + console
        sys.stdout = Logger(os.path.join(args.output_dir, "console_log.txt"))

    # Handle checkpoint directory
    if not args.checkpoint_dir and args.output_dir:
        args.checkpoint_dir = args.output_dir # Default to output_dir if not specified
    
    if args.checkpoint_dir and misc.is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(", ", ",\n"))
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    print(f"[Info] Logs will be saved to: {args.output_dir}")
    print(f"[Info] Checkpoints will be saved to: {args.checkpoint_dir}")

    device = torch.device(args.device)

    if args.wandb and misc.is_main_process():
        wandb.init(
            project="ct_vecset",
            name="test_run",
            config=args
        )
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True
    cudnn.benchmark = False # NOTE: disable for speed
    torch.use_deterministic_algorithms(True) # NOTE: disable for speed
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # NOTE: disable for speed

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # dataset_train = Objaverse(split="train", sdf_sampling=True, sdf_size=1024, surface_sampling=True, surface_size=args.point_cloud_size)
    # dataset_val = Objaverse(split="val", sdf_sampling=True, sdf_size=1024, surface_sampling=True, surface_size=args.point_cloud_size)

    dataset_train = CTSingleVolumeDataset(nii_path=args.data_path, pc_size=args.point_cloud_size, structure_intensity_threshold=args.structure_intensity_threshold)  # DUMMY TEST
    # Use same dataset for val for overfitting test
    dataset_val = dataset_train

    if args.wandb and misc.is_main_process():
        # Calculate coverage safely
        coverage = args.point_cloud_size / max(1, dataset_train.num_structure)
        
        wandb.log({
            "data/total_voxels": dataset_train.total_voxels,
            "data/structure_voxels": dataset_train.num_structure,
            "data/structure_ratio": dataset_train.structure_ratio,
            "data/encoding_vox/num_structure": coverage,
            "data/pc_size": args.point_cloud_size
        })
        print(f"[WandB] Logged structure stats. Coverage: {coverage*100:.2f}%")
        

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # prefetch_factor=2,
    )

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=sampler_val,
    #     # batch_size=args.batch_size,
    #     batch_size=1,
    #     # num_workers=args.num_workers,
    #     num_workers=1,
    #     pin_memory=args.pin_mem,
    #     drop_last=False
    # )

    # model = autoencoder.__dict__[args.model](pc_size=args.point_cloud_size)
    model = autoencoder.__dict__[args.model](pc_size=args.point_cloud_size, **vars(args))

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    criterion = torch.nn.L1Loss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
    #     exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # test_stats = evaluate(data_loader_val, model, device)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, log_writer=log_writer, args=args)
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # if epoch % 5 == 0 or epoch + 1 == args.epochs:
        #     # test_stats = evaluate(data_loader_val, model, device)

        #     # print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
        #     # max_iou = max(max_iou, test_stats["iou"])
        #     # print(f'Max iou: {max_iou:.2f}%')

        #     # if log_writer is not None:
        #     #     # log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
        #     #     log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                     # **{f'test_{k}': v for k, v in test_stats.items()},
        #                     'epoch': epoch,
        #                     'n_parameters': n_parameters}
        # else:
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if args.wandb and misc.is_main_process():
        wandb.finish()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args = load_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
