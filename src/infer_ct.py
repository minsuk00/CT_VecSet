# Copyright (c) 2025, Biao Zhang.
# [NOTE] Completely rewritten for CT Intensity Regression (Volume-to-Volume)

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchio as tio
import nibabel as nib
from models import autoencoder

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="point_vec1024x32_dim1024_depth24_nb", type=str, metavar="MODEL", help="Name of model architecture")
parser.add_argument("--pth", default="output/ae/checkpoint-140.pth", type=str, help="Path to checkpoint")
parser.add_argument("--input", type=str, required=True, help="Path to input .nii.gz file")
parser.add_argument("--output", type=str, required=True, help="Path to save output .nii.gz file")
parser.add_argument("--resolution", type=int, default=128, help="Resolution for reconstruction grid (e.g. 128, 256)")
parser.add_argument("--pc_size", type=int, default=8192, help="Number of context points for encoder")
parser.add_argument("--device", default="cuda", help="Device to use")
parser.add_argument("--seed", default=1, type=int)

def main():
    args = parser.parse_args()
    print(f"[Infer] Args: {args}")

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    device = torch.device(args.device)

    # 1. Initialize Model
    print(f"[Infer] Loading model: {args.model}")
    try:
        # Pass pc_size if model requires it
        model = autoencoder.__dict__[args.model](pc_size=args.pc_size)
    except KeyError:
        print(f"Model {args.model} not found in autoencoder.py")
        return

    # Load Checkpoint
    print(f"[Infer] Loading weights from: {args.pth}")
    checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    
    # Handle state dict (sometimes nested in 'model' key)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[Infer] Warning: Strict load failed ({e}). Trying strict=False.")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # 2. Load and Preprocess Data (Must match Training Logic)
    print(f"[Infer] Loading input volume: {args.input}")
    
    # [NOTE] Using same transforms as ct_dataset.py to ensure domain match
    transforms = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1), # Resample to 1mm iso
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
    ])

    subject = tio.Subject(ct=tio.ScalarImage(args.input))
    processed_subject = transforms(subject)
    data = processed_subject.ct.data # Tensor (1, D, H, W)
    
    D, H, W = data.shape[1:]
    print(f"[Infer] Volume Shape: {D}x{H}x{W}")

    # 3. Prepare Encoder Input (Structure Points)
    # [NOTE] Sample 'pc_size' points where intensity > 0.05
    structure_indices = torch.nonzero(data[0] > 0.05)
    
    if len(structure_indices) > args.pc_size:
        choice = torch.randint(0, len(structure_indices), (args.pc_size,))
        indices = structure_indices[choice]
    else:
        # Fallback: Sample randomly if structure is too small or empty
        print("[Infer] Warning: Not enough structure points. Sampling randomly.")
        z_idx = torch.randint(0, D, (args.pc_size,))
        y_idx = torch.randint(0, H, (args.pc_size,))
        x_idx = torch.randint(0, W, (args.pc_size,))
        indices = torch.stack([z_idx, y_idx, x_idx], dim=-1)

    # Normalize Coordinates [-1, 1]
    # Training logic assumes data is (D, H, W) -> (Z, Y, X)
    pc_z = (indices[:, 0] / (D - 1)) * 2 - 1
    pc_y = (indices[:, 1] / (H - 1)) * 2 - 1
    pc_x = (indices[:, 2] / (W - 1)) * 2 - 1
    
    # Model expects [x, y, z] order
    coords = torch.stack([pc_x, pc_y, pc_z], dim=-1).float() # (N, 3)

    # Extract Intensity Values
    vals = data[0, indices[:, 0], indices[:, 1], indices[:, 2]]
    vals = vals.unsqueeze(-1).float() # (N, 1)

    # [NOTE] Concatenate: (N, 4) -> [x, y, z, intensity]
    pc_input = torch.cat([coords, vals], dim=-1).unsqueeze(0).to(device) # (1, N, 4)
    
    print(f"[Infer] Encoder Input Shape: {pc_input.shape}")

    # 4. Prepare Decoder Query (Dense Grid)
    print(f"[Infer] Generating query grid with resolution {args.resolution}...")
    
    res = args.resolution
    z_range = torch.linspace(-1, 1, res)
    y_range = torch.linspace(-1, 1, res)
    x_range = torch.linspace(-1, 1, res)
    
    # Create grid with (Z, Y, X) layout using 'ij' indexing
    # This ensures that when we reshape to (res, res, res), the dimensions map to (Z, Y, X)
    grid_z, grid_y, grid_x = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
    
    # Stack in (X, Y, Z) order to match model's expected coordinate system
    query_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, -1, 3).to(device)
    
    print(f"[Infer] Query Shape: {query_points.shape}")

    # 5. Inference
    print("[Infer] Running Model...")
    with torch.no_grad():
        # [NOTE] model.forward handles chunking via block_size, so OOM is handled
        # pc_input: (1, N, 4)
        # query_points: (1, Res^3, 3)
        out_dict = model(pc_input, query_points, block_size=200000)
        predictions = out_dict["o"] # (1, N_queries)

    # 6. Save Output
    print("[Infer] Saving output...")
    # Reshape back to volume (Z, Y, X)
    vol_np = predictions.reshape(res, res, res).cpu().numpy()
    
    # Save as NIfTI
    save_path = args.output
    if not save_path.endswith('.nii.gz'):
        save_path += '.nii.gz'
        
    # Use identity affine. 
    # (Future improvement: Copy affine from input and scale it based on resolution change)
    nii_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
    nib.save(nii_img, save_path)
    
    print(f"[Infer] Saved reconstructed volume to: {save_path}")

if __name__ == "__main__":
    main()