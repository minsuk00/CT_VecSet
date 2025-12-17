# Copyright (c) 2025, Biao Zhang.

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils.lr_sched as lr_sched
import utils.misc as misc
from numpy import inner
from tqdm import tqdm
import wandb
import numpy as np

# 출력값(output)과 정답값(labels)을 특정 threshold 기준으로 이진화하여 IoU (Intersection over Union)와 정확도(Accuracy)를 계산
# It ignores how dense the tissue is. It only measures if the model correctly put "something" where "something" exists.
# Good for checking if the model learned the geometry/shape of the organ, even if the density values are slightly wrong.
def calc_iou(output, labels, threshold):
    target = torch.zeros_like(labels)
    target[labels >= threshold] = 1

    pred = torch.zeros_like(output)
    pred[output >= threshold] = 1

    accuracy = (pred == target).float().sum(dim=1) / target.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).gt(0).sum(dim=1) + 1e-5
    iou = intersection * 1.0 / union
    iou = iou.mean()
    return iou


# 경사도는 SDF(Signed Distance Function)의 경우 표면 법선 벡터를 의미. 이 경사도는 Eikonal Loss를 계산하는 데 필수
def points_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True)[0]
    return points_grad

# NEW: Helper function to visualize 3 slices
def visualize_slice(model, pc, device, resolution=256, gt_volume=None):
    model.eval()
    x = torch.linspace(-1, 1, resolution, device=device)
    y = torch.linspace(-1, 1, resolution, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') 
    
    # Normalized range is [-1, 1].
    # We pick 3 distinct levels
    # z_levels_norm = [-0.5, 0.0, 0.5]
    z_levels_norm = [-0.35, 0.0, 0.35]
    
    imgs_pred = []
    imgs_gt = []
    slice_indices = []

    with torch.no_grad():
        for z_val in z_levels_norm:
            # 1. Calculate the actual integer index (0 to resolution-1)
            # Map [-1, 1] -> [0, 1] -> [0, resolution-1]
            idx = int(round((z_val + 1) / 2 * (resolution - 1)))
            slice_indices.append(idx)

            # 2. Create grid for this Z slice
            grid_z = torch.full_like(grid_x, z_val)
            queries = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, -1, 3)
            
            # 3. Predict
            out = model(pc[:1], queries)
            pred = out["o"]
            
            # 4. Reshape to image
            img_pred = pred.reshape(resolution, resolution).cpu().numpy()
            
            # [DEBUG] Check range to debug black images
            # print(f"DEBUG: Slice {idx} min={img_pred.min():.4f}, max={img_pred.max():.4f}")
            
            # [NOTE] Clip to [0, 1] range to avoid negative values appearing black or scaling issues
            img_pred = np.clip(img_pred, 0, 1)
            imgs_pred.append(img_pred)

            # [NOTE] Sample Ground Truth if volume is provided
            if gt_volume is not None:
                # gt_volume is (1, D, H, W). grid_sample needs (N, C, D_in, H_in, W_in)
                # Queries are (1, N_pixels, 3). Reshape to (1, 1, H_out, W_out, 3)
                # Move grid to same device as gt_volume (likely CPU) to avoid large transfer
                tgt_device = gt_volume.device
                grid_gt = queries.view(1, 1, resolution, resolution, 3).to(tgt_device)
                
                input_gt = gt_volume.unsqueeze(0) # Add batch dim: (1, 1, D, H, W)
                
                # Sample
                sampled_gt = F.grid_sample(input_gt, grid_gt, align_corners=True) # (1, 1, 1, H, W)
                img_gt = sampled_gt.squeeze().cpu().numpy()
                img_gt = np.clip(img_gt, 0, 1) # Clip GT as well
                imgs_gt.append(img_gt)
            
    # [NOTE] Helper to concatenate images horizontally with separator
    def concat_row(img_list):
        if not img_list: return None
        sep_width = 10 # 10 pixels of white space
        # Assuming images are in [0, 1] range. 1.0 is white.
        separator = np.ones((resolution, sep_width), dtype=img_list[0].dtype)
        
        imgs_with_sep = []
        for i, img in enumerate(img_list):
            imgs_with_sep.append(img)
            if i < len(img_list) - 1: # Don't add after the last image
                imgs_with_sep.append(separator)
        return np.concatenate(imgs_with_sep, axis=1)

    # Build Rows
    row_pred = concat_row(imgs_pred)
    
    if gt_volume is not None and imgs_gt:
        row_gt = concat_row(imgs_gt)
        
        # [NOTE] Stack Vertically: GT on Top, Pred on Bottom
        # Add vertical separator
        sep_height = 10
        width = row_pred.shape[1]
        v_separator = np.ones((sep_height, width), dtype=row_pred.dtype)
        
        combined_img = np.concatenate([row_gt, v_separator, row_pred], axis=0)
    else:
        combined_img = row_pred

    model.train()
    # Return image, list of indices, and total count
    return combined_img, slice_indices, resolution

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.L1Loss()

    # if log_writer is not None:
    #     print("log_dir: {}".format(log_writer.log_dir))

    # for data_iter_step, (points, labels, structure_points, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    pbar = tqdm(data_loader, desc=header, disable=not misc.is_main_process())
    for data_iter_step, (points, labels, structure_points) in enumerate(pbar):
    # for data_iter_step, (points, labels, structure_points) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True) # near points + random points
        labels = labels.to(device, non_blocking=True)
        structure_points = structure_points.to(device, non_blocking=True) # zero-level set
        # print(points.shape) # torch.Size([1, 4096, 3])
        # print(labels.shape) # torch.Size([1, 4096, 1])
        # print(structure_points.shape) # torch.Size([1, 8192, 3])
        # surface_normals = surface_normals.to(device, non_blocking=True)

        # points: Volume Samples (물체 내부의 점들).
        # surface: Surface Samples (물체 표면의 점들).
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            points = points.requires_grad_(True)
            # points_all = torch.cat([points, structure_points], dim=1)  # points_all: 이 두 가지 샘플을 모두 합쳐 인코더/디코더에 입력.
            # outputs = model(structure_points, points_all)
            outputs = model(structure_points, points)
            # structure_points: 인코더 입력 -> Latent Code 생성
            # points_all: 디코더 입력. 값을 예측해야 할 좌표들(x,y,z) -> 출력값 생성
            # training efficient하게 할려고 points_all을 grid의 subset으로 함. surface가 중요한 부분이니까 일부러 포함
            output = outputs["o"]
            if output.dim() == 2:
                output = output.unsqueeze(-1)

            # grad = points_gradient(points_all, output)
            # TODO: CHANGE LOSS FOR CT
            # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # TODO: hard coded point numbers
                # loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()  # TODO: CT에서는 Eikonal Loss를 사용하지 않음
                # loss_vol = criterion(output[:, :1024], labels[:, :1024])  # Volume 내부의 샘플(1024개)에 대한 주된 재구성 손실입니다.
                # loss_near = criterion(output[:, 1024:2048], labels[:, 1024:2048])  # 표면 근처의 샘플(1024개)에 대한 손실입니다. 표면 경계는 학습하기 어려우므로 가중치 10을 주어 강조
                # loss_surface = (
                #     (output[:, 2048:]).abs().mean()
                # )  # surface 샘플(표면 점)의 예측값 절댓값을 최소화. 이는 SDF 모델에서 "표면에서의 거리는 0이 되어야 한다"($SDF=0)는 조건을 강제. TODO: CT에서는 사용 안함(?)

                # print(grad.shape, surface_normals.shape)
                # inner = torch.einsum('b n c, b n c -> b n', grad[:, 2048:], surface_normals)

                # print(inner.max(), inner.min(), inner.mean())
                # print(F.l1_loss(grad[:, 2048:], surface_normals), F.l1_loss(grad[:, 2048:], -surface_normals))
                # loss_surface_normal = F.l1_loss(F.normalize(grad[:, 2048:], dim=2), surface_normals)
                # loss_surface_normal = 1 - torch.einsum('b n c, b n c -> b n', (F.normalize(grad[:, 2048:], dim=2, eps=1e-6), surface_normals)).mean()

                # num_queries = points.shape[1]
                # output_valid = output[:, :num_queries] # Shape: (B, 4096)
                # # Unsqueeze to match labels shape (B, 4096, 1)
                # output_valid = output_valid.unsqueeze(-1) 
                # loss = criterion(output_valid, labels)
                # loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface  # + 0.01 * loss_surface_normal
            
            num_queries = points.shape[1]
            split_idx = num_queries // 2
            
            # Uniform (Empty Space Learning)
            output_uniform = output[:, :split_idx]
            labels_uniform = labels[:, :split_idx]
            loss_uniform = criterion(output_uniform, labels_uniform)
            
            # Structure (Detail Learning)
            output_struct = output[:, split_idx:]
            labels_struct = labels[:, split_idx:]
            loss_struct = criterion(output_struct, labels_struct)
            
            # [NEW] Weighted Sum: Emphasize Structure by 5x
            # loss = loss_uniform + 10.0 * loss_struct
            loss = loss_uniform + 2.0 * loss_struct
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)  # Backprop해줌
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # metric calculation
        with torch.no_grad():
            # MSE & PSNR
            # Since data is [0, 1], PSNR = -10 * log10(MSE)
            mse_val = F.mse_loss(output, labels).item() # best: 0.0, worst: 1.0
            psnr_val = -10.0 * math.log10(mse_val + 1e-10) # 1e-10 for numerical stability. best: 100, worst: 0
            
            # IoU with threshold 0.1 (Occupancy check)
            # Threshold 0.1: Is it structure? (>0.1) or Air? (<0.1)
            iou_val = calc_iou(output, labels, threshold=0.1) # best: 1.0, worst: 0.0

        # Update Loggers
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_uniform=loss_uniform.item())
        metric_logger.update(loss_struct=loss_struct.item())
        metric_logger.update(mse=mse_val)
        metric_logger.update(psnr=psnr_val)
        metric_logger.update(iou=iou_val)

        # metric_logger.update(loss_vol=loss_vol.item())
        # metric_logger.update(loss_near=loss_near.item())
        # metric_logger.update(loss_eikonal=loss_eikonal.item())
        # metric_logger.update(loss_surface=loss_surface.item())
        # metric_logger.update(loss_surface_normal=loss_surface_normal.item())

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        
        # Display Split Loss in Progress Bar
        pbar.set_postfix({
            "L_uni": f"{loss_uniform.item():.4f}", 
            "L_str": f"{loss_struct.item():.4f}", 
            "psnr": f"{psnr_val:.2f}"
        })

        if args and args.wandb and misc.is_main_process():
            wandb.log({
                "train/loss": loss_value,
                "train/loss_uniform": loss_uniform.item(),
                "train/loss_structure": loss_struct.item(),
                "train/mse": mse_val,
                "train/psnr": psnr_val,
                "train/iou": iou_val,
                "train/lr": max_lr,
                "epoch": epoch
            })
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # Visualize intermediate slice at the end of the epoch
    if args and args.wandb and misc.is_main_process():
        # Retrieve GT volume from dataset if available
        gt_volume = None
        if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'data'):
             gt_volume = data_loader.dataset.data
        img, slice_indices, total_slices = visualize_slice(model, structure_points, device, gt_volume=gt_volume)
        caption_str = f"Epoch {epoch} | Top: GT, Bottom: Pred | Slices: {slice_indices} / {total_slices}"
        wandb.log({"val/slices": [wandb.Image(img, caption=caption_str)]})
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
