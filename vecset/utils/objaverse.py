import csv
import os

import numpy as np
import torch
import trimesh
from torch.utils import data


# 3D 모델 파일(여기서는 .npz 파일)을 불러와, 모델이 학습하기 좋은 방식으로 좌표(Point)와 정답 값(SDF) 쌍으로 샘플링하여 텐서로 변환
class Objaverse(data.Dataset):
    def __init__(
        self,
        split,
        transform=None,
        sdf_sampling=True,
        sdf_size=4096,
        surface_sampling=True,
        surface_size=2048,
        dataset_folder="/ibex/project/c2281/objaverse",
        return_sdf=True,
    ):
        self.surface_size = surface_size

        self.transform = transform
        self.sdf_sampling = sdf_sampling
        self.sdf_size = sdf_size
        self.split = split

        self.surface_sampling = surface_sampling

        self.npz_folder = dataset_folder
        self.normal_folder = dataset_folder.replace("objaverse", "objaverse_normals")

        with open("utils/objaverse_{}.csv".format(split), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            model_filenames = [(os.path.join(self.npz_folder, row[0], row[1] + ".npz"), row[2]) for row in reader]
        self.models = model_filenames

        self.return_sdf = return_sdf

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        npz_path = self.models[idx][0]
        try:
            # SDF 값과 포인트 종류(Volume, Near, Surface)를 미리 계산되어 있음 .npz 파일에
            with np.load(npz_path) as data:
                vol_points = data["vol_points"]
                vol_sdf = data["vol_sdf"]
                near_points = data["near_points"]
                near_sdf = data["near_sdf"]
                surface = data["surface_points"]
        except Exception as e:
            idx = np.random.randint(self.__len__())
            return self.__getitem__(idx)

        # 원본 3D 모델의 표면 점(surface)이 너무 많을 경우, self.surface_size (기본 2048개)만큼 무작위로 점을 뽑아 *인코더의 입력(pc)*으로 사용될 텐서를 준비
        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.surface_size, replace=False)
            surface = surface[ind]
            # surface_normals = surface_normals[ind]
            # surface_normals = trimesh.unitize(surface_normals)
        surface = torch.from_numpy(surface)

        # surface_normals = torch.from_numpy(surface_normals).float()

        # 균형 샘플링 (vol_sdf < 0): SDF가 음수(내부)인 점과 SDF가 양수(외부)인 점의 개수가 비슷하도록 강제
        if self.sdf_sampling:
            ### make sure balanced sampling, maybe not necessary when doing sdf regression
            pos_vol_id = vol_sdf < 0

            if pos_vol_id.sum() > self.sdf_size // 2:
                ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size // 2, replace=False)
                pos_vol_points = vol_points[pos_vol_id][ind]
                pos_vol_sdf = vol_sdf[pos_vol_id][ind]
            else:
                pos_vol_id = near_sdf < 0
                if pos_vol_id.sum() > self.sdf_size // 2:
                    ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size // 2, replace=False)
                    pos_vol_points = near_points[pos_vol_id][ind]
                    pos_vol_sdf = near_sdf[pos_vol_id][ind]
                else:
                    ind = np.random.default_rng().choice(vol_points.shape[0], self.sdf_size // 2, replace=False)
                pos_vol_points = vol_points[ind]
                pos_vol_sdf = vol_sdf[ind]

            neg_vol_id = vol_sdf >= 0

            ind = np.random.default_rng().choice(neg_vol_id.sum(), self.sdf_size // 2, replace=False)
            neg_vol_points = vol_points[neg_vol_id][ind]
            neg_vol_sdf = vol_sdf[neg_vol_id][ind]

            vol_points = np.concatenate([pos_vol_points, neg_vol_points], axis=0)
            vol_sdf = np.concatenate([pos_vol_sdf, neg_vol_sdf], axis=0)
            ###

            ind = np.random.default_rng().choice(near_points.shape[0], self.sdf_size, replace=False)
            near_points = near_points[ind]
            near_sdf = near_sdf[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_sdf = torch.from_numpy(vol_sdf).float()

        if self.split == "train":
            near_points = torch.from_numpy(near_points)
            near_sdf = torch.from_numpy(near_sdf).float()

            points = torch.cat([vol_points, near_points], dim=0)
            sdf = torch.cat([vol_sdf, near_sdf], dim=0)
        else:
            near_points = torch.from_numpy(near_points)
            near_sdf = torch.from_numpy(near_sdf).float()

            points = near_points
            sdf = near_sdf
            # 검증/테스트 시에는 가장 중요한 영역(near_points)만 사용하여, 모델이 얼마나 디테일하고 정확하게 경계를 포착했는지 엄격하게 평가

        if self.transform:
            surface, points = self.transform(surface, points)

        ## random rotation
        if self.split == "train":
            perm = torch.randperm(3)
            points = points[:, perm]
            surface = surface[:, perm]

            negative = torch.randint(2, size=(3,)) * 2 - 1
            points *= negative[None]
            surface *= negative[None]

            roll = torch.randn(1)
            yaw = torch.randn(1)
            pitch = torch.randn(1)

            tensor_0 = torch.zeros(1)
            tensor_1 = torch.ones(1)

            RX = torch.stack(
                [torch.stack([tensor_1, tensor_0, tensor_0]), torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]), torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]
            ).reshape(3, 3)

            RY = torch.stack(
                [torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]), torch.stack([tensor_0, tensor_1, tensor_0]), torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]
            ).reshape(3, 3)

            RZ = torch.stack([torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]), torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]), torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(
                3, 3
            )

            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)

            points = torch.mm(points, R).detach()
            surface = torch.mm(surface, R).detach()
            # surface_normals = torch.mm(surface_normals, R).detach()

        if self.return_sdf is False:  # return occupancies (sign) instead
            sdf = (sdf < 0).float()

        return points, sdf, surface, self.models[idx][1], npz_path  # , surface_normals
