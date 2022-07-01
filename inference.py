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

# Reference - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

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
        return {"image": TC(Image.fromarray(io.imread(img_name))), "filename":self.filenames[idx]}

def getHeatmap(image, moduleList, num):
    '''
    names = []
    outputs = []
    output_ims = []

    params = list(model.fc.parameters())
    weight = np.squeeze(params[0].data.numpy())
    print('weight.shape', weight.shape)

    feature_maps = model.conv(image) # get the feature maps of the last convolutional layer
    print('feature maps shape', feature_maps.shape)

    layer = moduleList[-1]
    image = layer(image)
    names.append(str(layer))
    outputs.append(image)

    img = np.array(image)
    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406
    img = np.transpose(img, axes=[2, 1, 0])

    plt.imshow(img)
    plt.savefig('cam.jpeg')
    '''
    outputs = []
    names = []
    orig_image = image.unsqueeze(0)

    print("LEN", len(moduleList))
    for layer in moduleList[1:]:
        print('SHAPE', orig_image.shape)
        try:
            image = layer(orig_image)
        except:
            continue
        outputs.append(image)
        names.append(str(layer))
        
    output_im = []
    for i in outputs:
        i = i.squeeze(0)
        #print(i.shape)
        temp = torch.sum(i, dim=0)
        temp = torch.div(temp, temp.shape[0])
        output_im.append(temp.data.numpy())
        
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (80, 90)

    for i in range(len(output_im)):
        try:
            a = fig.add_subplot(8,4,i+1)
        except:
            continue
        tmpimg =  cv2.resize(np.array(output_im[i]),(224,224))
        imgplot = plt.imshow(tmpimg)
        plt.axis('off')
        a.set_title(names[i].partition('(')[0], fontsize=30)

    #UNCOMMENT THIS LINE TO SEE FIGURE
    #plt.savefig('./layer_outputs'+str(num)+'.jpg', bbox_inches='tight')

def inference():
    model = torchvision.models.mobilenet_v3_large(pretrained=True)

    last_channel = 1280
    lastconv_output_channels = 960
    num_classes = 3

    model.classifier[3] = nn.Linear(last_channel,num_classes)

    model.load_state_dict(torch.load("./models/2022-06-28/mobilenet_v3_large_finetuned.pt"))
    model.eval()

    data = StubDataset(dir="/Users/nagrawal/Documents/SmartPrimer/Smart Primer User Testing Location Photos")

    classes = ["background","eucalyptus","tree"]
    moduleList = list(model.features.modules())
    print("last element", moduleList[-1])
    '''
    print("MODULE LIST", len(moduleList))
    print("FIRST ELEMENT", moduleList[0])
    print()
    print("SECOND ELEMENT", moduleList[0])
    '''
    for i in range(len(data)):
        data_item = data[i]
        if data_item != None:
            image = data_item["image"]
            getHeatmap(image, moduleList, i)
            filename = data_item["filename"]
            print(filename + "," + classes[model(image.resize(1,3,224,224)).argmax()])

inference()



