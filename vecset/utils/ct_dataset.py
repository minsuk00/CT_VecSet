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
        self.structure_indices = torch.nonzero(self.data[0] > 0.05)

    def __len__(self):
        # Infinite length for "Iterative" training on single volume,
        # or a fixed number like 1000 to define an 'epoch'
        return 1000

    def __getitem__(self, idx):
        # --- 1. Encoder Input (pc) ---
        # "Context": What defines this CT scan?
        # We sample N points from the "Body" (non-zero regions)
        if len(self.structure_indices) > self.pc_size:
            choice = torch.randint(0, len(self.structure_indices), (self.pc_size,))
            indices = self.structure_indices[choice]
        else:
            # Fallback if volume is mostly empty
            indices = torch.randint(0, self.data.numel(), (self.pc_size,))
            # Convert flat indices to coords (omitted for brevity, assume structure_indices works)

        # Convert array indices (D, H, W) to normalized coordinates [-1, 1]
        # TorchIO/PyTorch GridSample uses [-1, 1] where -1 is left/top/front
        D, H, W = self.data.shape[1:]

        # Normalize Z, Y, X to [-1, 1]
        # indicies is [N, 3] -> (z, y, x) usually in torch
        pc_z = (indices[:, 0] / (D - 1)) * 2 - 1
        pc_y = (indices[:, 1] / (H - 1)) * 2 - 1
        pc_x = (indices[:, 2] / (W - 1)) * 2 - 1
        # VecSet expects [N, 3] input
        pc = torch.stack([pc_x, pc_y, pc_z], dim=-1).float()  # (N, 3)

        # --- 2. Decoder Query (points, labels) ---
        # "Training Data": Random points in continuous space
        # Random coordinates in [-1, 1]
        queries = torch.rand(self.query_size, 3) * 2 - 1  # (M, 3) range [-1, 1]
        # Get Ground Truth Intensity at these query points
        # We need 5D tensor for grid_sample: (1, C, D, H, W)
        # grid needs to be (1, 1, 1, M, 3) for 3D sampling trick or just (1, M, 1, 1, 3)
        input_tensor = self.data.unsqueeze(0)  # (1, 1, D, H, W)
        # grid_sample expects (x, y, z) in last dim
        grid = queries.view(1, -1, 1, 1, 3)
        sampled_intensity = F.grid_sample(input_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True)  # Output: (1, 1, M, 1, 1)
        labels = sampled_intensity.view(-1, 1)  # (M, 1)

        # Return format: (Decoder_Input, Decoder_Target, Encoder_Input)
        return queries.float(), labels.float(), pc.float()
