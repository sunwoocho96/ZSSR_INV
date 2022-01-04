import os
import argparse
import torch
import sys
import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.utils import read_image, write_image, kernel_shift

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from model import ZSSR
from config.configs import Config
from dataloader.dataloader import DataGenerator

def train(kernel, conf):
    print('-' * 30 + '\nRunning ZSSR X%d...' % (conf.sf))
    print('image : ', conf.input_img_path)
    start_time = time.time()
    self.kernel = kernel_shift(kernel, sf)

    self.input = read_image(conf.input_img_path)
    self.son_input = father_to_son(self.input, self.kernel)

    zssr = ZSSR(conf, self.input, self.son_input)
    data = DataGenerator(conf, self.input, self.kernel)
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)

    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [lr, gt] = data.__getitem__(iteration)
        loss, sr = zssr.train(lr, gt, iteration)

    self.output = zssr.test(self.input)
    write_image(conf.output_img_path, self.output)
    runtime = int(time.time() - start_time)

    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)



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
