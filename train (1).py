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
from functions import get_input_args, load_and_transform, build_classifier, train, save_model

''' 
This file let's the user train a pre-trained neural network on an image dataset.
The user can choose the learning rate, epochs, model architecture, input features, hidden layers,
output_nodes and whether he/she wants to use GPU to train it.
'''
def main():
    # keep track of training time
    start_time = time.time()
    # user inputs from command line
    in_arg = get_input_args()
    # load and process data into training, validation and test data sets
    trainloader, validationloader, testloader, train_data = load_and_transform(in_arg.data_dir)
    # build classifier with user inputs (loss criterion & optimizer are fixed)
    model, criterion, optimizer = build_classifier(in_arg.arch, in_arg.in_features, in_arg.hidden_layers, in_arg.output_nodes, in_arg.learning_rate)
    # Train the model
    training_model = train(in_arg.epochs, trainloader, validationloader, optimizer, model, criterion, in_arg.gpu)        
    # Save Checkpoint
    saved_model = save_model(model, optimizer, in_arg.saving_dir, in_arg.arch, in_arg.hidden_layers, in_arg.in_features, in_arg.learning_rate, in_arg.epochs, in_arg.output_nodes, train_data)
# Call to main function to run the program
if __name__ == "__main__":
    main()