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
    
    # [NOTE] Use GT resolution if available, otherwise default
    res_d, res_h, res_w = resolution, resolution, resolution
    if gt_volume is not None:
        # gt_volume shape is (1, D, H, W)
        res_d, res_h, res_w = gt_volume.shape[1:]
        # print(f"DEBUG: Using GT resolution: D={res_d}, H={res_h}, W={res_w}")

    # Base linspace for each dimension
    # We need separate grids for D, H, W because they might be different
    ticks_d = torch.linspace(-1, 1, res_d, device=device)
    ticks_h = torch.linspace(-1, 1, res_h, device=device)
    ticks_w = torch.linspace(-1, 1, res_w, device=device)
    
    # Normalized positions for slices (25%, 50%, 75%)
    slice_pos_norm = [-0.25, 0.0, 0.25]
    
    # Store images for all directions
    # Structure: [Axial_Row, Coronal_Row, Sagittal_Row]
    all_direction_imgs = []
    
    with torch.no_grad():
        # Iterate over 3 directions: 0=X (Sagittal), 1=Y (Coronal), 2=Z (Axial)
        for direction in [2, 1, 0]: 
            imgs_pred_row = []
            imgs_gt_row = []
            
            # Construct grids based on the *plane* we are visualizing
            if direction == 2:   # Axial (Z-plane): Fix Z, vary X, Y. Grid is (H, W)
                u, v = torch.meshgrid(ticks_h, ticks_w, indexing='ij') 
                res_u, res_v = res_h, res_w
            elif direction == 1: # Coronal (Y-plane): Fix Y, vary X, Z. Grid is (D, W)
                u, v = torch.meshgrid(ticks_d, ticks_w, indexing='ij')
                res_u, res_v = res_d, res_w
            else:                # Sagittal (X-plane): Fix X, vary Y, Z. Grid is (D, H)
                u, v = torch.meshgrid(ticks_d, ticks_h, indexing='ij')
                res_u, res_v = res_d, res_h

            for pos in slice_pos_norm:
                fixed = torch.full_like(u, pos)
                
                # Construct 3D query based on direction
                if direction == 2:   # Fix Z (Axial): (x, y, fixed) -> x=v(W), y=u(H)
                    queries = torch.stack([v, u, fixed], dim=-1) 
                elif direction == 1: # Fix Y (Coronal): (x, fixed, z) -> x=v(W), z=u(D)
                    queries = torch.stack([v, fixed, u], dim=-1) 
                else:                # Fix X (Sagittal): (fixed, y, z) -> y=v(H), z=u(D)
                    queries = torch.stack([fixed, v, u], dim=-1)

                queries = queries.reshape(1, -1, 3) # (1, N, 3)
                
                # Predict
                out = model(pc[:1], queries)
                img_pred = out["o"].reshape(res_u, res_v).cpu().numpy()
                # img_pred = torch.sigmoid(out["o"]).reshape(res_u, res_v).cpu().numpy() # NOTE: L1 -> BCE
                img_pred = np.clip(img_pred, 0, 1)
                imgs_pred_row.append(img_pred)

                # Sample GT
                if gt_volume is not None:
                    tgt_device = gt_volume.device
                    # Grid for sampling must be (1, 1, H_out, W_out, 3)
                    grid_gt = queries.view(1, 1, res_u, res_v, 3).to(tgt_device)
                    input_gt = gt_volume.unsqueeze(0) 
                    sampled_gt = F.grid_sample(input_gt, grid_gt, align_corners=True) 
                    img_gt = sampled_gt.squeeze().cpu().numpy()
                    img_gt = np.clip(img_gt, 0, 1)
                    imgs_gt_row.append(img_gt)

            # Concatenate this row (3 slices)
            row_pred = np.concatenate(imgs_pred_row, axis=1) # [Slice1 | Slice2 | Slice3]
            
            # If GT exists, stack GT on top of Pred for this direction
            if gt_volume is not None and imgs_gt_row:
                row_gt = np.concatenate(imgs_gt_row, axis=1)
                sep_h = np.ones((5, row_pred.shape[1]), dtype=row_pred.dtype) # Thin separator
                full_row = np.concatenate([row_gt, sep_h, row_pred], axis=0)
            else:
                full_row = row_pred
            
            all_direction_imgs.append(full_row)

    # Now stack the 3 direction blocks vertically. 
    # Note: They might have different widths if D != H != W. 
    # We need to pad them to the max width to concatenate vertically.
    max_width = max(img.shape[1] for img in all_direction_imgs)
    
    padded_imgs = []
    for img in all_direction_imgs:
        if img.shape[1] < max_width:
            pad_width = max_width - img.shape[1]
            # Pad right side with white (1.0)
            padding = np.ones((img.shape[0], pad_width), dtype=img.dtype)
            img = np.concatenate([img, padding], axis=1)
        padded_imgs.append(img)

    final_img = padded_imgs[0]
    sep_block = np.ones((20, max_width), dtype=final_img.dtype) # Thick separator between views
    
    for i in range(1, len(padded_imgs)):
        final_img = np.concatenate([final_img, sep_block, padded_imgs[i]], axis=0)

    model.train()
    return final_img, slice_pos_norm

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

            # output_probs = torch.sigmoid(output) # NOTE: L1 -> BCE
            # criterion = torch.nn.BCEWithLogitsLoss() # NOTE: L1 -> BCE
            
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
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)  # Backprop해줌
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # metric calculation
        with torch.no_grad():
            # MSE & PSNR
            # Since data is [0, 1], PSNR = -10 * log10(MSE)
            # mse_val = F.mse_loss(output_probs, labels).item() # best: 0.0, worst: 1.0 # NOTE: L1 -> BCE
            mse_val = F.mse_loss(output, labels).item() # best: 0.0, worst: 1.0
            psnr_val = -10.0 * math.log10(mse_val + 1e-10) # 1e-10 for numerical stability. best: 100, worst: 0
            
            # IoU with threshold 0.1 (Occupancy check)
            # Threshold 0.1: Is it structure? (>0.1) or Air? (<0.1)
            # iou_val = calc_iou(output_probs, labels, threshold=0.1) # best: 1.0, worst: 0.0
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
                "epoch": epoch,
                "train/grad_norm": grad_norm,
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
    if args and args.wandb and misc.is_main_process() and (epoch % args.vis_iter == 0):
        gt_volume = None
        if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'data'):
             gt_volume = data_loader.dataset.data
        
        img, positions = visualize_slice(model, structure_points, device, gt_volume=gt_volume)
        caption_str = f"Epoch {epoch} | Top: GT, Bottom: Pred | Views: Axial (Top), Coronal (Mid), Sagittal (Bot)"
        wandb.log({"val/multi_view": [wandb.Image(img, caption=caption_str)]})
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
