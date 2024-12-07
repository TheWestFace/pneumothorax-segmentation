import os
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class PneumoDataset(Dataset):
    def __init__(self, mode, fold_index=None, train_names=None, test_names=None,
                 fold_labels=None, transform=None):
        
        self.transform = transform
        
        # change to your path
        self.train_image_path = 'data/1img_mask/images/'
        self.train_mask_path = 'data/1img_mask/mask/'
        self.train_image_name = train_names
        self.test_image_path = 'data/3test_imgs/'
        self.test_image_name = test_names 
        
        self.fold_labels = fold_labels
        self.set_mode(mode, fold_index)
        self.to_tensor = ToTensorV2()

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            # get train list image names
            self.train_list = self.train_image_name[self.fold_labels != fold_index].tolist()
            self.num_data = len(self.train_list)

        elif self.mode == 'val':
            # get val list image names
            self.val_list = self.train_image_name[self.fold_labels == fold_index].tolist()
            self.num_data = len(self.val_list)

        elif self.mode == 'test':
            self.test_list = self.test_image_name[self.fold_labels == fold_index].tolist()
            self.num_data = len(self.test_list)

    def __getitem__(self, index):
        if self.mode == 'test':
            image = cv2.imread(os.path.join(self.test_image_path, self.test_list[index]), 1)
            sample = {"image": image}
            if self.transform:
                sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
            image = sample['image']
            image_id = self.test_list[index].replace('.png', '')
            return image_id, image
        
        elif self.mode == 'train':
            image = cv2.imread(os.path.join(self.train_image_path, self.train_list[index]), 1)
            label = cv2.imread(os.path.join(self.train_mask_path, self.train_list[index]), 0) / 255.

        elif self.mode == 'val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index]), 1)
            label = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index]), 0) / 255.

        sample = {"image": image, "mask": label}
        if self.transform:
            sample = self.transform(**sample)
        sample = self.to_tensor(**sample)
        image, label = sample['image'].to(torch.float32), sample['mask'].unsqueeze(0).to(torch.float32)
            
        return image, label
         
    def __len__(self):
        return self.num_data
    
from torch.utils.data.sampler import Sampler
class PneumoSampler(Sampler):
    def __init__(self, fold_index, demand_non_empty_proba, 
                 fold_labels, exist_labels):
        assert demand_non_empty_proba > 0, 'frequency of non-empty images must be greater then zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba
        self.train_exist = exist_labels[fold_labels != fold_index]

        self.positive_idxs = np.where(self.train_exist == 1)[0]
        self.negative_idxs = np.where(self.train_exist == 0)[0]

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        
    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative