import argparse
import torch
import os
import scipy.io as sio
import numpy as np


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # self.parser.add_argument('--input_img_path', default=os.path.dirname(__file__) + '/input/input.png')
        # self.parser.add_argument('--output_img_path', default=os.path.dirname(__file__) + '/output/output.png')
        self.parser.add_argument('--input_dir', default=os.path.dirname(__file__) + '/input/')
        self.parser.add_argument('--output_dir', default=os.path.dirname(__file__) + '/output/')

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
