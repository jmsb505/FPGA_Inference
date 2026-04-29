import glob
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TucoDataset(Dataset):
    def __init__(self, data_root, split='train', normalize=True):
        self.data_root = data_root
        self.split = split
        self.normalize = normalize
        self.volumes_dir = os.path.join(data_root, 'volumes_center')
        self.seg_dir = os.path.join(data_root, 'seg_center')

        if not os.path.isdir(self.volumes_dir):
            raise FileNotFoundError(f"Volumes directory not found: {self.volumes_dir}")

        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(f"No MR/CT pairs found for split '{split}' in {self.volumes_dir}")

        print(f"Found {len(self.pairs)} volume pairs for split '{split}'")

    def _find_pairs(self):
        all_files = glob.glob(os.path.join(self.volumes_dir, '*.nii.gz'))
        mr_files = [path for path in all_files if path.endswith('_mr.nii.gz')]

        pairs = []
        for mr_path in mr_files:
            subject_id = os.path.basename(mr_path).replace('_mr.nii.gz', '')
            ct_path = os.path.join(self.volumes_dir, f'{subject_id}_ct.nii.gz')
            if not os.path.exists(ct_path):
                continue

            mr_seg_path = os.path.join(self.seg_dir, f'{subject_id}_mr.nii.gz')
            ct_seg_path = os.path.join(self.seg_dir, f'{subject_id}_ct.nii.gz')

            if not os.path.exists(mr_seg_path) and not os.path.exists(ct_seg_path):
                common_seg_path = os.path.join(self.seg_dir, f'{subject_id}.nii.gz')
                if os.path.exists(common_seg_path):
                    mr_seg_path = common_seg_path
                    ct_seg_path = common_seg_path
                else:
                    mr_seg_path = None
                    ct_seg_path = None
            else:
                if not os.path.exists(mr_seg_path):
                    mr_seg_path = None
                if not os.path.exists(ct_seg_path):
                    ct_seg_path = None

            pairs.append({
                'id': subject_id,
                'mr_path': mr_path,
                'ct_path': ct_path,
                'mr_seg_path': mr_seg_path,
                'ct_seg_path': ct_seg_path,
            })

        pairs.sort(key=lambda pair: pair['id'])
        n_val = max(1, int(len(pairs) * 0.1))
        n_test = max(1, int(len(pairs) * 0.1))

        if self.split == 'val':
            return pairs[:n_val]
        if self.split == 'test':
            return pairs[n_val:n_val + n_test]
        return pairs[n_val + n_test:]

    def _load_volume(self, path):
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        if self.normalize:
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        return torch.from_numpy(data).unsqueeze(0)

    def _load_seg(self, path):
        if path is None:
            return None
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        return torch.from_numpy(data).unsqueeze(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        moving = self._load_volume(pair['mr_path'])
        fixed = self._load_volume(pair['ct_path'])
        moving_seg = self._load_seg(pair['mr_seg_path'])
        fixed_seg = self._load_seg(pair['ct_seg_path'])

        sample = {
            'moving': moving,
            'fixed': fixed,
            'id': pair['id'],
            'meta': {
                'mr_path': pair['mr_path'],
                'ct_path': pair['ct_path'],
            },
        }
        if moving_seg is not None:
            sample['moving_seg'] = moving_seg
        if fixed_seg is not None:
            sample['fixed_seg'] = fixed_seg
        return sample
