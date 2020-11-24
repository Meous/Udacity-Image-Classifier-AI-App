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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Path to the image to be predicted')
parser.add_argument('--checkpoint', type=str, help='Path to a saved model')
parser.add_argument('--top_k', type=int, help='Top K predictions')
parser.add_argument('--class_name', type=str, help='JSON file that has mapping for flower names')
parser.add_argument('--gpu', action='store_true', help='Use GPU')

args, _ = parser.parse_known_args()

image = args.image if args.image else ''
gpu = args.gpu if args.gpu else False
top_k = args.top_k if args.top_k else 5
class_name = args.class_name if args.class_name else ''
checkpoint = args.checkpoint if args.checkpoint else ''

def load_checkpoint(path):
    checkpoint_path = torch.load(path)
    
    if checkpoint_path['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint_path['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('{} is unspported by current application'.format(arch))
        sys.exit() 
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint_path['class_to_idx']
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint_path['input_size'], checkpoint_path['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint_path['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    # Replace classifier
    model.classifier = classifier
    model.load_state_dict(checkpoint_path['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model    
    pil_image = Image.open(image)
    # Resize to 256 
    if pil_image.width > pil_image.height:
        pil_image.thumbnail((10000000,256))
    else:
        pil_image.thumbnail((256,10000000))
    
    # Crop to 224 * 224
    left = (pil_image.width-224)/2
    bottom = (pil_image.height-224)/2
    right = left + 224
    top = bottom + 224

    pil_image = pil_image.crop((left, bottom, right,top))
    
    np_image = np.array(pil_image)/255   
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image
def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # Process image to get numpy array
    np_image = process_image(image_path)
    # Convert nump array to tensor
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    # Add batch of size 1 to image
    tensor_image.unsqueeze_(0)

    tensor_image = tensor_image.to(device)
    model.to(device)
    model.eval()
    output = model.forward(tensor_image)
    # Probabilities
    ps = torch.exp(output)

    # Top five probs
    top_prob, top_class = ps.topk(top_k)
    if device == torch.device('cuda:0'):
        top_prob, top_class = top_prob.cpu(), top_class.cpu()
    
    probs = top_prob.detach().numpy().tolist()[0] 
    class_codes = top_class.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[cls] for cls in class_codes]
    return probs, classes
cat_to_name = None
if class_name:
    with open(class_name, 'r') as f:
        cat_to_name = json.load(f)
if checkpoint:    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    model = load_checkpoint(checkpoint)
    probs, classes = predict(image, model, top_k)
    flower_names = []
    if cat_to_name:
        for cls in classes:
            flower_names.append(cat_to_name[cls])
    print("Prediction results for {}".format(image))
    print("Top {} probabilities: {}".format(top_k, probs))
    print("Top {} classes: {}".format(top_k, classes))
    print("Top {} flowers: {}".format(top_k, flower_names))