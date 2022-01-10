import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append(('../'))
from utils.utils import read_image, create_gradient_map, im2tensor, create_probability_map, father_to_son

class DataGenerator(Dataset):


    def __init__(self, conf, input, kernel):

        self.conf = conf
        self.input = input
        self.kernel = kernel
        self.shave_edges(sf=conf.sf)

        self.in_rows, self.in_cols = self.input.shape[0:2]

        self.crop_indices = self.make_list_of_crop_indices(conf=conf)


    def __len__(self):
        return self.crop_indices.shape[0]


    def __getitem__(self,idx):

        gt = self.next_crop(idx=idx)
        lr = father_to_son(gt, kernel=self.kernel, sf=self.conf.sf)


        gt = im2tensor(gt)
        lr = im2tensor(lr)

        return lr, gt


    def shave_edges(self, sf):
        self.input = self.input[10:-10, 10:-10, :]

        shape = self.input.shape
        self.input = self.input[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input
        self.input = self.input[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input


    def make_list_of_crop_indices(self, conf):
        iteration = conf.max_iters
        prob_map = self.create_prob_maps(scale_factor=conf.sf)
        crop_indices = np.random.choice(a=len(prob_map), size=iteration, p=prob_map)

        return crop_indices

    def create_prob_maps(self, scale_factor):
        loss_map = create_gradient_map(self.input)
        prob_map = create_probability_map(loss_map, self.conf.patch_size)

        return prob_map

    def next_crop(self, idx):
        size = self.conf.patch_size

        top, left = self.get_top_left(size, idx)
        crop_im = self.input[top: top + size, left:left + size, :]
        crop_im += np.random.randn(*crop_im.shape) / 255.0

        return crop_im

    def get_top_left(self, size, idx):
        center = self.crop_indices[idx]
        row, col = int(center/self.in_cols), center % self.in_cols
        top, left = min(max(0, row-size//2), self.in_rows - size), min(max(0, col - size //2), self.in_cols - size)

        return top - top %2 , left - left % 2
