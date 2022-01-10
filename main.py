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
from utils.utils import read_image, write_image, kernel_shift, father_to_son, im2tensor, tensor2im

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model.ZSSR import ZSSR
from config.config import Config
from dataloader.dataloader import DataGenerator

def train(conf):
    print('-' * 30 + '\nRunning ZSSR X%d...' % (conf.sf))
    print('image : ', conf.input_img_path)
    start_time = time.time()

    input = read_image(conf.input_img_path)

    if conf.kernel_path != 'bic':
        kernel = kernel_shift(load(conf.kernel_path)['Kernel'], conf.sf)
    else :
        kernel = None
    son_input = father_to_son(hr_father=input, kernel=kernel, sf=conf.sf)
    son_input = im2tensor(son_input)

    data = DataGenerator(conf, input, kernel)
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)

    input = im2tensor(input)
    zssr = ZSSR(conf, input, son_input)
    output = tensor2im(input)
    write_image(output, conf.output_img_path)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [lr, gt] = data.__getitem__(iteration)
        loss, sr = zssr.train(lr, gt, iteration)

    output = zssr.test(input)
    output = tensor2im(output)
    write_image(output, conf.output_img_path)
    runtime = int(time.time() - start_time)

    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)


def create_params(filename, args):
    kernel_path = os.path.join(args.input_dir, filename[:-4]+'.mat')
    
    kernel_path = kernel_path if os.path.exists(kernel_path) else 'bic'
    params = ['--input_img_path', os.path.join(args.input_dir, filename),
              '--output_img_path', os.path.join(args.output_dir, filename),
              '--kernel_path', kernel_path]

    return params

def main():

    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default='ZSSR')
    prog.add_argument('--input_dir', type=str, default='input')
    prog.add_argument('--output_dir', type=str, default='output')

    args = prog.parse_args()

    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()

    for filename in filesource[:]:
        print(filename)
        if args.model == 'ZSSR' :
            conf = Config().parse(create_params(filename, args))
            train(conf)
    prog.exit(0)

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
