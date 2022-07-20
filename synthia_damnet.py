import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model.model_builder import init_model
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
from trainer.damnet_trainer import Trainer
import copy
import numpy as np
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()

def main():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    config, writer = init_config('./config/synthia_damnet_config.yml', sys.argv)

    config.num_classes = 19

    model = init_model(config)

    trainer = Trainer(model, config, writer)

    trainer.train()

if __name__ == "__main__":
    main()
