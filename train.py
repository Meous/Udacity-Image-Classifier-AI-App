import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import time
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Directory to train, valid and test subfolders')
parser.add_argument('--arch', type=str, help='Neural Network architecture')
parser.add_argument('--gpu', action='store_true', help='Set training to GPU')
parser.add_argument('--epochs', type=int, help='Epochs for model training')
parser.add_argument('--hidden_units', type=int, help='Hidden units in the classifier')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--checkpoint', type=str, help='Checkpoint for saving model')

args, _ = parser.parse_known_args()

data_dir = args.data_dir if args.data_dir else ''
gpu = args.gpu if args.gpu else False
epochs = args.epochs if args.epochs else 5
arch = args.arch if args.arch else 'densenet121'
learning_rate = args.learning_rate if args.learning_rate else 0.001
hidden_units = args.hidden_units if args.hidden_units else 500
checkpoint = args.checkpoint if args.checkpoint else ''
    
def validation(model, valid_loader, criterion):
    valid_loss = 0
    valid_accuracy = 0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss/len(valid_loader), valid_accuracy/len(valid_loader)

if data_dir:
    #Load Data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_image_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_image_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=64, shuffle=False)
    
    # Define Model
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        print('{} not supported'.format(arch))
        sys.exit()
    # Freeze parameters
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Define classifier
    classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(input_size, hidden_units)),
                           ('relu1', nn.ReLU()),
                           ('dropout',nn.Dropout(0.3)),
                           ('fc2', nn.Linear(hidden_units, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))
    # Replace pre-trained classifier with the new one
    model.classifier = classifier
    #print (model)
    # Define device, loss function and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 40
    steps = 0
    running_loss = 0
    # change to cuda
    model.to(device)
    
    for e in range(epochs):
        model.train()
        for images, labels in train_dataloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)      
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, valid_dataloader, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy))
                running_loss = 0
                # Make sure training is back on
                model.train()
    if checkpoint:          
        model.class_to_idx = train_image_dataset.class_to_idx
        torch.save({'arch': arch,
                    'state_dict': model.state_dict(), 
                    'class_to_idx': model.class_to_idx,
                    'epochs': epochs,
                    'hidden_units': hidden_units,
                    'input_size': input_size}, 
                    checkpoint)