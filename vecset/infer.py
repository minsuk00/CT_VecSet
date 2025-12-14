# Copyright (c) 2025, Biao Zhang.

# import yaml
# import math
import argparse
from pathlib import Path

import mcubes
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
import trimesh
import utils.misc as misc
from models import autoencoder
from scipy.spatial import (
    cKDTree as KDTree,
)
from tqdm import tqdm
from utils.objaverse import (
    Objaverse,
)
from utils.shapenet import (
    ShapeNet,
    category_ids,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="point_vec1024x32_dim1024_depth24_nb", type=str, metavar="MODEL", help="Name of model to train")
parser.add_argument("--pth", default="output/ae/point_vec1024x32_dim1024_depth24_sdf/checkpoint-140.pth", type=str)
parser.add_argument("--device", default="cuda", help="device to use for training / testing")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--pc_size", type=int, default=8192)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()


def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = autoencoder.__dict__[args.model](pc_size=args.pc_size)
    device = torch.device(args.device)

    model.eval()
    model.load_state_dict(torch.load(args.pth, map_location="cpu", weights_only=False)["model"], strict=True)
    model.to(device)

    density = args.resolution
    gap = 2.0 / density
    x = np.linspace(-1, 1, density + 1)
    y = np.linspace(-1, 1, density + 1)
    z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
    # view: (3, N^3)
    # transpose: (N^3, 3) (i.e., each row is x,y,z coord)
    # None: (1, N^3, 3)

    with torch.no_grad():
        surface = trimesh.load(args.input).vertices.astype(np.float32)
        shifts = (surface.max(axis=0) + surface.min(axis=0)) / 2
        surface = surface - shifts
        distances = np.linalg.norm(surface, axis=1)
        scale = 1 / np.max(distances)
        surface *= scale

        ind = np.random.default_rng().choice(surface.shape[0], args.pc_size, replace=False)
        surface = surface[ind]
        surface = torch.from_numpy(surface)[None].to(device)

        outputs = model(surface, grid)["o"][0]
        volume = outputs.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()  # * (-1)

        verts, faces = mcubes.marching_cubes(volume, 0)
        verts *= gap
        verts -= 1.0
        m = trimesh.Trimesh(verts, faces)
        m.export(args.output)


if __name__ == "__main__":
    main()
