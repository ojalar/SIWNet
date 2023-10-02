import numpy as np
import itertools
import argparse
import torch
import torchvision.transforms as transforms

from base import create_base_net
from pihead import PIHead
from siwnet import SIWNet

def train(params_path, train_path, val_path, test_path = None, save_path = None, name = None):
    # function for training SIWNet
    
    # establish training transforms for image preprocessing
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomRotation((-4, 4)),
        transforms.Resize((324, 324)),
        transforms.ToTensor(),
        transforms.Normalize((0.39497562, 0.37916522, 0.35401782), 
                (0.12773657,  0.12749068, 0.12992096))
    ])
    
    # initialise base net (feature backbone + point estimate head)
    base_net = create_base_net()
    # initialise prediction interval head
    pi_head = PIHead(input_size=513)
    # initialise SIWNet
    siwnet = SIWNet(base_net, pi_head, save_path, name)

    # if no path to test file provided, train on training data and evaluate on validation data
    if test_path is None:
        siwnet.train(params_path, [train_path], [val_path], train_transforms)
    # if a test path is provided, train on combined training and validation data, evaluate on
    # test data
    else:
        siwnet.train(params_path, [train_path, val_path], [test_path], train_transforms)


if __name__ == "__main__":
    torch.manual_seed(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", help="path to params file", required = True)
    ap.add_argument("-tr", "--train", help="path to training file", required = True)
    ap.add_argument("-v", "--val", help="path to validation file", required = True)
    ap.add_argument("-te", "--test", help="path to test file")
    ap.add_argument("-s", "--save", help="path to directory for saving models and results")
    ap.add_argument("-n", "--name", help="name of the trained model")
    args = vars(ap.parse_args())
    print(args)
    train(args["params"], args["train"], args["val"], args["test"], args["save"], args["name"])
