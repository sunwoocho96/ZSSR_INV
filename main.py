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


def create_params(filename, args):
    params = ['--input_img_path', os.path.join(args.input_dir, filename),
              '--output_img_path', os.path.join(args.output_dir, filename)]

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
