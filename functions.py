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
from collections import OrderedDict
from pathlib import Path

'''
This .py file contains the functions responsible for training a neural network and making new predictions on unseen data.
'''

# keep track of training time
start_time = time.time() 

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    This function returns these arguments as an ArgumentParser object.
    """
    ## Define command line arguements for user 

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments using add_argument() from ArguementParser method
    # start with arguments used to train the model
    parser.add_argument('--data_dir', type = str, default = "flowers",
                         help='Please enter path to folder of training and test data of images')
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                        help='Please enter a CNN model architecture: "vgg16" or "alexnet"')
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Please specify a learning rate. (default = 0.001")
    parser.add_argument("--in_features", type = int, default = 25088, 
                        help="Please enter the number of hidden layers as list format. For vgg choose 25088 (default) and for alexnet choose 9216")
    parser.add_argument("--hidden_layers", type = int, default = 512, 
                        help="Please enter the number of hidden layers as list format. (Default = [25088, 512]")
    parser.add_argument("--output_size", type = int, default = 102,
                        help="Please enter the number of output nodes. (Default = 102)")
    parser.add_argument("--epochs", type = int, default = 3,
                        help="Please enter the number of epochs to train the model. (default = 3)")
    parser.add_argument("--gpu", type = bool, default = True,
                        help="Please specify if you would like to use GPU or CPU to train the model (True/False). (Default = True)")
    parser.add_argument("--saving_dir", type = str, default = "checkpoint_1.pth",
                        help="Please enter a file path to specify where to save the trained model and its hyperparameters.")
    # arguments used to make predictions
    parser.add_argument("--checkpoint", type = str, default = "checkpoint_1.pth",
                        help="Please enter a model checkpoint you would like to load. (Default = checkpoint_1.pth)")
    parser.add_argument("--image_path", type = str, default = "flowers/test/51/image_01362.jpg",
                        help="Please enter a file path to a new image that you would like to use for prediction. (Default = flowers/test/51/image_01362.jpg)")
    parser.add_argument("--topk", type = int, default = 5,
                        help="Please enter the number of the top most probable classes you would like to display. (Default = 5)")   
    return parser.parse_args()

# load and process input images
def load_and_transform(data_dir): # default is flowers dataset
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

## transformations for the training, validation, and testing sets
# training data: apply transformations random scaling, cropping, flipping, data is resized to 224x224 pixels (data augmentation)
    traindata_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224), 
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

# validation data: no scaling or rotation but resize then crop images to appropriate size
    validationdata_transforms = transforms.Compose([transforms.Resize(256), # resize to square
                                                    transforms.CenterCrop(224),  # crops a square out of the center
                                                    transforms.ToTensor(), # convert to tensor to be able to use in my network
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])])

# test data: no scaling or rotation but resize then crop images to appropriate size
    testdata_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=traindata_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validationdata_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=testdata_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    print("Input data was loaded and processed succesfully.")
    
    return trainloader, validationloader, testloader, train_data

## After loading and processing the data, the user can build his/her model

# define classifier specified by user input
def build_classifier(arch, in_features, hidden_layers, output_size, learning_rate): 
    if arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
         # freeze pre-trained model parameters
        for param in model.parameters():
            param.requires_grad = False
    elif arch == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
         # freeze pre-trained model parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry this model is not available. Please choose between vgg16 and alexnet.")
                    
   
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_layers)), # arch and hidden layers specified by user
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.5)), # dropout fixed at 0.5
                                            ('fc2', nn.Linear(hidden_layers, 128)),
                                            ('relu', nn.ReLU()),
                                            ('dropout2',nn.Dropout(0.5)),
                                            ('fc3', nn.Linear(128, output_size)), # output defined by user
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    model.classifier = classifier
    # combine pre-trained nn with user-built classifier 
    model.classifier = nn.Sequential(nn.Linear(in_features, hidden_layers),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_layers, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(128, output_size),
                                     nn.LogSoftmax(dim=1))
    # define loss criterion    
    criterion = nn.NLLLoss() #NLLLoss because of LogSoftMax
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)   #learning rate specified by user
    print("Model was built successfully.")
    return model, criterion, optimizer


## Training the model
def train(epochs, trainloader, validationloader, optimizer, model, criterion, gpu):
    ''' 
    Trains the model. The user can specify the number of epochs and whether to use GPU or CPU.
    '''
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #Use GPU if it's available
    if gpu == True:
        model.to(device)
    else:
        model.to("cpu")
    epochs = epochs 
    steps = 0 # track number of training steps
    running_loss = 0 
    print_every = 15 

# loop through image data
    print("Starting training...")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1 # increment steps by 1
            start_time = time.time() 
            # Move input and label tensors to the default device (GPU if available)
            inputs, labels = inputs.to(device), labels.to(device)
            # reset existing gradients
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward() # backward pass
            optimizer.step()

            running_loss += loss.item() # this way I can keep track of my running loss
            # drop out of loop because training loop done

            # set up and beginning of validation loop
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps) # get actualy probability distributions
                        top_p, top_class = ps.topk(1, dim=1) # get top probabilities and classes from ps.topk()
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # calculate accuracy from equals

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "  # shows average loss accross batches
                      f"Validation Loss: {validation_loss/len(validationloader):.3f}.. " # shows average loss accross batches
                      f"Model Accuracy: {accuracy/len(validationloader):.3f}") # shows average accuracy accross batches
        
                end_time = time.time() 
                tot_time = end_time - start_time
                print("\n** Total Elapsed Runtime:",
                str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
                +str(int((tot_time%3600)%60)),"\n")        
    return model

# saving the model   
def save_model(trained_model, optimizer, saving_dir, arch, learning_rate, epochs, train_data):
    ''' 
    Saves the trained model and its parameters for future use.
    '''
    trained_model.class_to_idx = train_data.class_to_idx 
    trained_model.to("cpu")
    checkpoint={'classifier': trained_model.classifier,
                'optimizer': optimizer,
                'architecture': arch, 
                'state_dict': trained_model.state_dict(),
                'class_to_idx': trained_model.class_to_idx,
                'learning_rate': learning_rate,
                'epochs': epochs}

    torch.save(checkpoint, saving_dir)
    
    print("Training complete and model is saved!" "\n" "You can now use the model to make predictions by entering 'python predict.py' in the command line.")   

# load the checkpoint and rebuild the model
def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    if checkpoint["architecture"] == "vgg16":
        model = getattr(torchvision.models, "vgg16")(pretrained=True)
    elif checkpoint["architecture"] == "alexnet":
        model = getattr(torchvision.models, "alexnet")(pretrained=True)
    else:
        print("Unknown model entered. Please make sure you saved the right architecture (vgg16 or alexnet).") 
        
    model.classifier = checkpoint['classifier'] # contains input nodes, hidden layers and output nodes
    model.optimizer = checkpoint['optimizer'] # in case we want to keep training the model
    model.load_state_dict(checkpoint['state_dict']) # state dict holds the weights for all layers including classifier so we need to assign classifier first and then load the model state dict
    model.class_to_idx = checkpoint['class_to_idx']
    model.learning_rate = checkpoint['learning_rate'] # in case we want to keep training the model
    model.epochs = checkpoint['epochs'] # in case we want to keep training the model
    print("Model was loaded successfully.")
    return model

# load and process unseen image data for prediction
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = PIL.Image.open(image_path)   # load image via PIL module

    image_transforms = transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) 
    pil_image = image_transforms(pil_image)   
    print("Image data was loaded and preprocessed successfully.")
    return pil_image

# make predictions on unseen data
def predict(new_image, model_checkpoint, topk, gpu):
    ''' 
    Predict the category of an image using a trained deep learning model.
    '''
    print("Making predictions on unseen image data...")
    # used this as reference: https://knowledge.udacity.com/questions/30810
    # Process the input image with earlier defined function  
    new_image = new_image.unsqueeze(0) # not sure why it's needed but using it just to be sure. 
    #found this here: https://knowledge.udacity.com/questions/30304
    new_image = new_image.float() # convert to float because model expects float. Source: https://knowledge.udacity.com/questions/246749
    model_checkpoint.eval() #set to eval() because we are not training the model anymore 
    if gpu == "True":
        model_checkpoint = model_checkpoint.to("cuda" if torch.cuda.is_available() else "cpu") # set both model and inputs (here: image) to "cuda" to use GPU
        new_image = new_image.to("cuda" if torch.cuda.is_available() else "cpu")     
    else:
        model_checkpoint = model_checkpoint.to("cpu") # set both model and inputs (here: image) to "cuda" to use GPU
        new_image = new_image.to("cpu")
    # using model to predict new image
    with torch.no_grad():
        logps = model_checkpoint.forward(new_image)
        ps = torch.exp(logps) # get actual probability distributions        
        probs, classes = ps.topk(topk, dim = 1) # get top k=5 largest values in a tensor from torch.topk()

        return probs, classes

# get labels
def get_labels(probs, classes, model_checkpoint):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    probs = probs.type(torch.FloatTensor).to('cpu').numpy()[0] # conver to numpy array 
    classes = classes.type(torch.FloatTensor).to('cpu').numpy()[0].tolist() # convert to list for compatability to get labels
    idx_to_class = {v: k for k, v in model_checkpoint.class_to_idx.items()} 
    flower_names = [idx_to_class[x] for x in classes]
    flower_names = [cat_to_name[str(x)] for x in flower_names]
    
    print("Predictions complete. \n\nResults: \n \nCategory: {} \nProbability: {}".format(flower_names, probs))
 
