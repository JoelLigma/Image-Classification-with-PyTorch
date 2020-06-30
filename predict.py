# Module Imports 
import numpy as np
import pandas as pd
import json
import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import PIL
from PIL import Image
import argparse 
# import defined functions from functions.py
import functions
from functions import get_input_args, process_image, load_checkpoint, predict, get_labels

''' 
This file let's the user employ a trained neural network to make predictions on unseen data.
.
'''
def main():
    # user inputs from command line
    in_arg = get_input_args()
    # load model checkpoint
    model_checkpoint = load_checkpoint(in_arg.checkpoint)
    # load and process unseen image data
    new_image = process_image(in_arg.image_path)
    # predict on unseen image data
    probs, classes = predict(new_image, model_checkpoint, in_arg.topk, in_arg.gpu) 
    # get labels
    get_labels(probs, classes, model_checkpoint)
# Call to main function to run the program
if __name__ == "__main__":
    main()

