import numpy as np
from torch.utils.data import Dataset
from .imresize import imresize
import sys
sys.path.append(('../'))
from utils.util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation

class DataGenerator(Dataset):
    def __init__(self,conf,gan):
        self.input_image = read_image(conf.input_image_path) / 255.
        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real)
        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)

