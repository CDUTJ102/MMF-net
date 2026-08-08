import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import os
from numpy import random


class CMRDataset(Dataset):
    def __init__(self, dir, mode='train', num_slices=3, slice_gap=1):
        self.mode = mode
        self.num_slices = num_slices
        self.slice_gap = slice_gap
        assert num_slices % 2 == 1, 'num_slices must be an odd number'
        assert slice_gap >= 1, 'slice_gap must be greater than or equal to 1'
        self.radius = num_slices // 2
        self.index_map = []
        self.ct_list = []
        self.oar_list = []
        self.ptv_list = []
        self.dose_list = []
        train_path = dir + 'train_set'
        test_path = dir + 'val_set'
        if self.mode == 'train':
            path = train_path
        else:
            path = test_path
        for pat in os.listdir(path):
            print(pat)
            CT = sitk.ReadImage(os.path.join(path, pat) + '/ct.nii.gz')
            oar = sitk.ReadImage(os.path.join(path, pat) + '/oars.nii.gz')
            ptv = sitk.ReadImage(os.path.join(path, pat) + '/ptvs.nii.gz')
            Dose = sitk.ReadImage(os.path.join(path, pat) + '/dose.nii.gz')

            CT = sitk.GetArrayFromImage(CT)
            oar = sitk.GetArrayFromImage(oar)
            ptv = sitk.GetArrayFromImage(ptv)
            Dose = sitk.GetArrayFromImage(Dose) * 72.6 / 80

            D = CT.shape[0]
            pad_z = self.radius * self.slice_gap
            pad_width = ((pad_z, pad_z), (0, 0), (0, 0))

            CT = np.pad(CT, pad_width, mode='constant', constant_values=0)
            oar = np.pad(oar, pad_width, mode='constant', constant_values=0)
            ptv = np.pad(ptv, pad_width, mode='constant', constant_values=0)
            Dose = np.pad(Dose, pad_width, mode='constant', constant_values=0)

            self.ct_list.append(CT.astype(np.float32))
            self.oar_list.append(oar.astype(np.float32))
            self.ptv_list.append(ptv.astype(np.float32))
            self.dose_list.append(Dose.astype(np.float32))

            pat_idx = len(self.ct_list) - 1
            for original_z in range(D):
                padded_z = original_z + pad_z
                self.index_map.append((pat_idx, padded_z))
        print('load done, length of dataset:', len(self.index_map))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        pat_idx, z = self.index_map[idx]
        z_list = [z + (i - self.radius) * self.slice_gap
                  for i in range(self.num_slices)]

        image = np.stack([self.ct_list[pat_idx][z_list],
                          self.oar_list[pat_idx][z_list],
                          self.ptv_list[pat_idx][z_list]], axis=0)
        label = self.dose_list[pat_idx][z:z + 1]

        tensor_image1 = torch.from_numpy(image)
        tensor_label = torch.from_numpy(label)
        if self.mode == 'train':
            tensor_image1, tensor_label = self.RandomFlip(tensor_image1, tensor_label)
            tensor_image1, tensor_label = self.RandomRotate90(tensor_image1, tensor_label)
        return tensor_image1, tensor_label

    def RandomFlip(self, img1, label, axis_prob=0.5, axis=-2):
        if random.uniform() > axis_prob:
            img1 = img1.numpy()
            label = label.numpy()
            if random.uniform() > axis_prob:
                axis = -1
            img1 = np.flip(img1, axis)
            label = np.flip(label, axis)
            img1 = torch.from_numpy(img1.copy())
            label = torch.from_numpy(label.copy())
        return img1, label

    def RandomRotate90(self, img1, label, axis_prob=0.5):
        if random.uniform() > axis_prob:
            img1 = img1.numpy()
            label = label.numpy()
            axis = (-2, -1)
            k = random.randint(0, 4)
            img1 = np.rot90(img1, k, axis)
            label = np.rot90(label, k, axis)
            img1 = torch.from_numpy(img1.copy())
            label = torch.from_numpy(label.copy())
        return img1, label


if __name__ == '__main__':
    trainset = CMRDataset(dir='./data/',  mode='train')
    print(trainset[10][0].shape)
    print(len(trainset))
