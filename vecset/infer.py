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
parser.add_argument("--model", default="point_vec1024x32_dim1024_depth24_nb", type=str, metavar="MODEL", help="사용할 아키텍처의 이름")
parser.add_argument("--pth", default="output/ae/point_vec1024x32_dim1024_depth24_sdf/checkpoint-140.pth", type=str, help="학습된 모델 가중치 파일 경로")

parser.add_argument("--input", type=str, required=True, help="재구성할 대상 3D 파일")
parser.add_argument("--output", type=str, required=True, help="재구성된 3D 파일의 출력 경로")
parser.add_argument(
    "--resolution", type=int, default=128, help="재구성된 3D 메쉬의 해상도. 128 x 128 x 128 크기의 볼륨을 만들어 Mesh를 추출. 자유롭게 설정할 수 있으며, 이 값이 재구성된 3D 모델의 디테일 수준을 결정"
)
parser.add_argument("--pc_size", type=int, default=8192, help="인코더의 입력으로 사용되는 Point Cloud의 샘플 개수")

parser.add_argument("--device", default="cuda", help="device to use for training / testing")
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
        # Loads a 3D mesh file (args.input) using the trimesh library and extracts only the vertices
        surface = trimesh.load(args.input).vertices.astype(np.float32)
        shifts = (surface.max(axis=0) + surface.min(axis=0)) / 2
        surface = surface - shifts  # Centers the object at the origin (0, 0, 0)
        distances = np.linalg.norm(surface, axis=1)  # Calculates the distance of every point from the new center (0, 0, 0)
        scale = 1 / np.max(distances)
        surface *= scale  # Scales the object to fit within the range [-1, 1]

        # Select a random subset of indices from the full point cloud.
        ind = np.random.default_rng().choice(surface.shape[0], args.pc_size, replace=False)
        surface = surface[ind]
        # Convert the final point cloud into a PyTorch tensor, and add the batch dimension [None]
        surface = torch.from_numpy(surface)[None].to(device)

        # feeds the (sub)sampled points (surface) and the dense coordinates (grid) into the neural network (model).
        # surface: 코더의 입력이 되어 Latent Code를 생성
        # grid: 디코더의 Query가 되어 출력값을 생성
        outputs = model(surface, grid)["o"][0]
        # Reshapes the flat $N^3$ output predictions back into a 3D cubic tensor
        # correct the axis order (e.g., X, Y, Z) to match the convention expected by the Marching Cubes implementation (mcubes).
        volume = outputs.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()  # * (-1)

        # Marching Cubes
        verts, faces = mcubes.marching_cubes(volume, 0)
        verts *= gap
        verts -= 1.0
        m = trimesh.Trimesh(verts, faces)
        m.export(args.output)


if __name__ == "__main__":
    main()
