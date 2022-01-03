import pytorch
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec
from configs import Config
from utils import *

class ZSSR:
    kernel = None
    learning_rate = None

    def __init__(self, input_img_path, sf=2, ground_truth_path=None, kernels=None, is_real_img=False, noise_scale=1.):
        self.conf = conf
        self.input = img.imread(input_img_path)
        if self.input.shape[-1] == 4 :
            self.input = img.imread(input_img_path)[:,:,:3]
        elif len(self.input.shape) == 2 :
            self.input = np.expand_dims(self.input, -1)

        self.gt = img.imread(ground_truth_path)

        self.kernels = [kernel_shift(kernel, sf) for kernel, sf in zip(kernels, self.conf.scale_factors)] if kernels is not None else [self.conf.downscale_method] * len(self.conf.scale_factors)
        

    def run(self):

        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):

            sf = [sf, sf] if np.isscalar(sf) else sf
            self.sf = np.array(sf) / np.array(self.base_sf)

            self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * sf ))

            self.init_sess(init_weights = self.conf.init_net_for_each_sf))

            self.train()

            post_processed_output = self.final_test()

            self.hr_fathers_sources.append(post_processed_output)

            self.loss_map_sources.append(create_loss_map(im=post_processed_output)) if self.conf.grad_based_loss_map else self.loss_map_sources.append(np.ones_like(post_processed_output))

            self.base_change()

        
        return post_processed_output


    def build_network(self.meta):
        

