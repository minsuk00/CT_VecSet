import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset


class CTSingleVolumeDataset(Dataset):
    def __init__(self, nii_path, pc_size=8192, query_size=4096, split="train"):
        """
        Args:
            nii_path: Path to the single .nii.gz file
            pc_size: Number of points for Encoder input (Condition)
            query_size: Number of points for Decoder query (Training)
        """
        self.pc_size = pc_size
        self.query_size = query_size
        self.split = split

        # 1. Load Data using TorchIO
        # We use a RescaleIntensity transform to ensure data is in [0, 1]
        # CT usually ranges -1000 to 1000+. We squash it to 0-1.
        self.transforms = tio.Compose(
            [
                tio.ToCanonical(),
                tio.Resample(1),  # Resample to 1mm iso (optional but recommended)
                tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),  # Robust normalization. Clip to 0,1
                tio.EnsureShapeMultiple(8),  # Pads or crops the volume so that the dimensions (D, H, W) are all multiples of 8.
            ]
        )

        # Create a Subject
        subject = tio.Subject(ct=tio.ScalarImage(nii_path))

        # Apply transforms once (since we are overfitting to a single fixed volume for now)
        # For real training, you would apply random transforms inside __getitem__
        self.processed_subject = self.transforms(subject)

        # Cache the data tensor [1, D, H, W]
        self.data = self.processed_subject.ct.data
        print(f"Volume Shape: {self.data.shape}")

        # Pre-calculate indices for the Encoder Input (Optimization)
        # We want the encoder to look at 'interesting' parts (e.g., bones, organs),
        # not just empty black space (0).
        # Strategy: Sample indices where intensity > 0.1
        # self.structure_indices = torch.nonzero(self.data[0] > 0.05)
        self.structure_indices = torch.nonzero(self.data[0] > 0.3)

        self.total_voxels = self.data.numel()
        self.num_structure = len(self.structure_indices)
        self.structure_ratio = self.num_structure / self.total_voxels
        
        print(f"[Dataset] Total Voxels: {self.total_voxels}") # 262144 for torch.Size([1, 64, 64, 64])
        print(f"[Dataset] Structure Voxels (>0.05): {self.num_structure}") # 14147
        print(f"[Dataset] Structure Ratio: {self.structure_ratio:.4f} ({self.structure_ratio*100:.2f}%)") # 0.0540 (5.40%)
        print(f"[Dataset] Encoding Voxels: {pc_size}") # 14147
        
        # [NEW] Calculate coverage: How much of the structure does the encoder see per pass?
        # pc_size is the number of points we sample per batch.
        coverage_per_pass = pc_size / max(1, self.num_structure)
        print(f"[Dataset] Encoder Coverage (Encoding Voxels N={pc_size}/Structure Voxels): {coverage_per_pass:.4f} ({coverage_per_pass*100:.2f}%)")
        coverage_per_pass = pc_size / max(1, self.total_voxels)
        print(f"[Dataset] Encoder Coverage (Encoding Voxels N={pc_size}/Total Voxels): {coverage_per_pass:.4f} ({coverage_per_pass*100:.2f}%)")

    def __len__(self):
        # Infinite length for "Iterative" training on single volume,
        # or a fixed number like 1000 to define an 'epoch'
        return 300

    def __getitem__(self, idx):
        D, H, W = self.data.shape[1:]
        
        # --- 1. Encoder Input (pc) ---
        # "Context": What defines this CT scan? Input to Encoder (and also give decoder points when training)
        # We sample N points from the "Body" (non-zero regions)
        if len(self.structure_indices) >= self.pc_size:
            choice = torch.randint(0, len(self.structure_indices), (self.pc_size,))
            indices = self.structure_indices[choice]
        else:
            print("Fallback. Sampling from entire volume instead")
            # Fallback if volume is mostly empty
            z_idx = torch.randint(0, D, (self.pc_size,))
            y_idx = torch.randint(0, H, (self.pc_size,))
            x_idx = torch.randint(0, W, (self.pc_size,))
            indices = torch.stack([z_idx, y_idx, x_idx], dim=-1)
        # indices = self.structure_indices #NOTE: FOR TESTING PURPOSE
        
        # Convert array indices (D, H, W) to normalized coordinates [-1, 1]
        # TorchIO/PyTorch GridSample uses [-1, 1] where -1 is left/top/front

        # Normalize Z, Y, X to [-1, 1]
        # indicies is [N, 3] -> (z, y, x) usually in torch
        pc_z = (indices[:, 0] / (D - 1)) * 2 - 1
        pc_y = (indices[:, 1] / (H - 1)) * 2 - 1
        pc_x = (indices[:, 2] / (W - 1)) * 2 - 1
        # VecSet expects [N, 3] input
        pc_coords = torch.stack([pc_x, pc_y, pc_z], dim=-1).float()  # (N, 3)

        # Get Encoder Intensities! 
        # We use the integer indices to lookup values
        pc_vals = self.data[0, indices[:, 0], indices[:, 1], indices[:, 2]].unsqueeze(-1).float()
        
        # Combine -> (N, 4) [x, y, z, intensity]
        pc_full = torch.cat([pc_coords, pc_vals], dim=-1)

        # --- 2. Decoder Query (points, labels) ---
        # "Training Data": Random points in continuous space
        # Random coordinates in [-1, 1]
        # queries = torch.rand(self.query_size, 3) * 2 - 1  # (M, 3) range [-1, 1]
        # # Get Ground Truth Intensity at these query points
        # # We need 5D tensor for grid_sample: (1, C, D, H, W)
        # # grid needs to be (1, 1, 1, M, 3) for 3D sampling trick or just (1, M, 1, 1, 3)
        # input_tensor = self.data.unsqueeze(0)  # (1, 1, D, H, W)
        # # grid_sample expects (x, y, z) in last dim
        # grid = queries.view(1, -1, 1, 1, 3)
        # sampled_intensity = F.grid_sample(input_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True)  # Output: (1, 1, M, 1, 1)
        # labels = sampled_intensity.view(-1, 1)  # (M, 1)

        
        # A. Uniform Queries (Learn Empty Space)
        n_uniform = self.query_size // 2
        q_uniform = torch.rand(n_uniform, 3) * 2 - 1
        
        # B. Structure Queries (Learn Details)
        n_struct = self.query_size - n_uniform
        if len(self.structure_indices) > 0:
            choice_q = torch.randint(0, len(self.structure_indices), (n_struct,))
            struct_ind = self.structure_indices[choice_q]
            
            # Normalize structure indices to coords [-1, 1]
            sz = (struct_ind[:, 0] / (D - 1)) * 2 - 1
            sy = (struct_ind[:, 1] / (H - 1)) * 2 - 1
            sx = (struct_ind[:, 2] / (W - 1)) * 2 - 1
            q_struct = torch.stack([sx, sy, sz], dim=-1).float()
        else:
            q_struct = torch.rand(n_struct, 3) * 2 - 1

        # Combine
        queries = torch.cat([q_uniform, q_struct], dim=0) # (TotalQuery, 3)
        
        # Get Labels for ALL queries using grid_sample
        input_tensor = self.data.unsqueeze(0) # (1, 1, D, H, W)
        grid = queries.view(1, -1, 1, 1, 3)
        
        sampled = F.grid_sample(input_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True)
        labels = sampled.view(-1, 1)

        # Return format: (Decoder_Input, Decoder_Target, Encoder_Input)
        return queries.float(), labels.float(), pc_full.float()
