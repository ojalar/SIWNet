import numpy as np
import itertools
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from base import create_base_net
from pihead import PIHead
from siwnet import SIWNet
from utils import plot_trunc_norm

def inference(path_wb, path_pi, img_path, save_path):
    # function to perform inference with SIWNet
    
    # initialise base network, PI head, and SIWNet
    base_net = create_base_net()
    pi_head = PIHead(input_size=513)
    siwnet = SIWNet(base_net, pi_head)
    
    # load weights from provided paths
    siwnet.load_weights(path_wb, path_pi)
    # define path for displaying result
    result_path = save_path + os.path.basename(img_path).split('.')[0] + ".pdf"

    # read provided image
    img = Image.open(img_path).convert("RGB")
    # perform inference
    pred, std = siwnet.inference(img)
    # acquire values from tensor output
    pred = pred.item()
    std = std.item()
    # plot prediction and define prediction interval
    pi = plot_trunc_norm(pred, std, name = result_path) 
    # print results
    print("Predicted friction factor:", pred)
    print("Predicted standard deviation:", std)
    print("Predicted 90% interval:", (pi[0], pi[1]))
    print("Result plotted to: " + result_path)

if __name__ == "__main__":
    torch.manual_seed(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("-wb", "--weights_base", help="path to weights of base net", required = True)
    ap.add_argument("-wp", "--weights_pi", help="path to weights of PI head", required = True)
    ap.add_argument("-i", "--image", help="path to image", required = True)
    ap.add_argument("-s", "--save", help="path to save directory", default = "")
    args = vars(ap.parse_args())
    print(args)
    inference(args["weights_base"], args["weights_pi"], args["image"], args["save"])
