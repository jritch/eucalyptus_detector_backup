from ast import Sub
import torch
import torchvision
import torch.nn as nn
from skimage import io
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Reference - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

classes = ["background","eucalyptus","tree"]

class StubDataset(Dataset):
    def __init__(self,dir):
        self.dir = dir
        self.filenames = os.listdir(dir)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.dir,self.filenames[idx])
        TC = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if (self.filenames[idx] == '.DS_Store'):
            return None
        return {"image": TC(Image.fromarray(io.imread(img_name))), "filename":self.filenames[idx], "img_name": img_name}

#Heatmap code adapted from: https://debuggercafe.com/traffic-sign-recognition-using-pytorch-and-deep-learning/
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # Generate the class activation maps upsample to 256x256.
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def apply_color_map(CAMs, width, height, orig_image):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        result = cv2.resize(result, (224, 224))
        return result

def visualize_and_save_map(
    result, orig_image, gt_idx=None, class_idx=None, save_name=None
):
    # Put class label text on the result.
    if class_idx is not None:
        cv2.putText(
            result, 
            f"Pred: {str(classes[int(class_idx)])}", (5, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            cv2.LINE_AA
        )
    if gt_idx is not None:
        cv2.putText(
            result, 
            f"GT: {str(classes[int(gt_idx)])}", (5, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            cv2.LINE_AA
        )
    # cv2.imshow('CAM', result/255.)
    orig_image = cv2.resize(orig_image, (224, 224))
    # cv2.imshow('Original image', orig_image)
    img_concat = cv2.hconcat([
        np.array(result, dtype=np.uint8), 
        np.array(orig_image, dtype=np.uint8)
    ])
    cv2.imshow('Result', img_concat)
    cv2.waitKey(1)
    if save_name is not None:
        cv2.imwrite(f"{save_name}.jpg", img_concat)

model = torchvision.models.mobilenet_v3_large(pretrained=True)

last_channel = 1280
lastconv_output_channels = 960
num_classes = 3

model.classifier[3] = nn.Linear(last_channel,num_classes)

model.load_state_dict(torch.load("./models/2022-06-28/mobilenet_v3_large_finetuned.pt"))
model.eval()

# Hook the feature extractor. This allows us to extract the model features. 
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('features').register_forward_hook(hook_feature)
# Get the softmax weight.
params = list(model.parameters())
print('model')
print(model)
#We take the softmax weights by skipping the last 4 layers which consist of the fully connected layers. 
#The layer where the weights are taken from is the pooling layer after the final 2D Convolutional layer. 
#This index will change for every model according to its architecture.
weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

print('softmax weights')
print(weight_softmax)
# Define the transforms, resize => tensor => normalize.
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def inference():

    #Note: Change this to your path
    data = StubDataset(dir="/Users/nagrawal/Documents/SmartPrimer/Smart Primer User Testing Location Photos")

    for i in range(len(data)):
        data_item = data[i]
        if data_item != None:
            image = data_item["image"]
            filename = data_item["filename"]
            print(filename + "," + classes[model(image.reshape(1,3,224,224)).argmax()])
            pred = model(image.reshape(1,3,224,224)).argmax()
            # Generate class activation mapping for the top1 prediction.
            CAMs = returnCAM(features_blobs[0], weight_softmax, pred)
            # File name to save the resulting CAM image with.
            save_name = './cam/cam'+str(i)
            # Show and save the results.
            image = cv2.imread(data_item["img_name"])
            orig_image = image.copy()
            height, width, _ = orig_image.shape
            result = apply_color_map(CAMs, width, height, orig_image)
            df = pd.read_csv("output.csv", index_col='filename')
            ground_truth = classes.index(df.loc[filename]['label'])
            visualize_and_save_map(result, orig_image, ground_truth, pred, save_name)

inference()



