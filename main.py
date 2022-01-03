import os
import argparse
import torch
import sys
import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from util import read_image, im2tensor01, map2tensor, tensor2im01, evaluation_dataset
# for KernelGAN-KP
from config.configs_FKP import Config_FKP
from dataloader.dataloader_FKP import DataGenerator_FKP
from model.model_FKP import KernelGAN_FKP
# for KernelGAN
from config.configs import Config
from dataloader.dataloader import DataGenerator
from model.model import KernelGAN
from model.learner import Learner
import time
# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from model import ZSSR


def run_zssr(k_2, conf):
    """Performs ZSSR with estimated kernel for wanted scale factor"""
    if conf.do_ZSSR:
        start_time = time.time()
        print('~' * 30 + '\nRunning ZSSR X%d...' % (4 if conf.X4 else 2))
        if conf.X4:
            sr = ZSSR(conf.input_image_path, scale_factor=[[2, 2], [4, 4]], kernels=[k_2, analytic_kernel(k_2)], is_real_img=conf.real_image, noise_scale=conf.noise_scale).run()
        else:
            sr = ZSSR(conf.input_image_path, scale_factor=2, kernels=[k_2], is_real_img=conf.real_image, noise_scale=conf.noise_scale).run()
        max_val = 255 if sr.dtype == 'uint8' else 1.
        plt.imsave(os.path.join(conf.output_dir_path, 'ZSSR_%s.png' % conf.img_name), sr, vmin=0, vmax=max_val, dpi=1)
        runtime = int(time.time() - start_time)
        print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)

def create_params(filename, args):
    params = ['--i', os.path.join(args.input_dir, filename),
              '--o', os.path.join(args.output_dir, filename)]

    return params

def main():

    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default='ZSSR_INV')

    args = prog.parse_args()

    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()

    for filename in filesource[:]:
        print(filename)
        if args.model == 'ZSSR' :
            conf = Config_ZSSR().parse(create_params(filename, args))
            kernelname = filename[:-4] + '.mat'
            kernel = load(kernel)['Kernel']
            run_zssr(kernel, conf)
    prog.exit(0)

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
