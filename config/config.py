import argparse
import torch
import os
import scipy.io as sio
import numpy as np

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # input, output image directory
        self.parser.add_argument('--input_img_path', default=os.path.dirname(__file__) + '/input/input.png')
        self.parser.add_argument('--output_img_path', default=os.path.dirname(__file__) + '/output/output.png')
        self.parser.add_argument('--input_dir', default=os.path.dirname(__file__) + '/input/')
        self.parser.add_argument('--output_dir', default=os.path.dirname(__file__) + '/output/')
        self.parser.add_argument('--kernel_path', default='bic')
        self.parser.add_argument('--patch_size', default= 64)



        ## training hyper-params
        self.parser.add_argument('--sf', default=2)
        self.parser.add_argument('--lr', default=0.001)
        self.parser.add_argument('--beta1', default=0.5)

        self.parser.add_argument('--learning_rate_policy_check_every', default=60)
        self.parser.add_argument('--learning_rate_slope_range', default=256)
        self.parser.add_argument('--run_test_every', default=50)
        self.parser.add_argument('--learning_rate_change_ratio', default= 1.5)

        self.parser.add_argument('--max_iters', default=3000)
        self.parser.add_argument('--min_iters', default=256)

        ## network architecture hyper-params

        self.parser.add_argument('--n_layers', default=8)
        self.parser.add_argument('--mid_channels', default=64)
        self.parser.add_argument('--inv_stride', default=1)

        ## gpu setting
        self.parser.add_argument('--gpu_id', default=0)


    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        print("Scale Factor: %s \t Input directory: %s" % (str(self.conf.sf), self.conf.input_dir))



        return self.conf


    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)

