from ast import Sub
import torch
import torchvision
import torch.nn as nn
from skimage import io
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 

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
        return {"image": TC(Image.fromarray(io.imread(img_name))), "filename":self.filenames[idx]}

model = torchvision.models.mobilenet_v3_large(pretrained=True)

last_channel = 1280
lastconv_output_channels = 960

model.classifier = nn.Sequential(
    nn.Linear(lastconv_output_channels, last_channel),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(last_channel, 3),
)

model.load_state_dict(torch.load("mobilenet_v3_large_finetuned.pt"))
model.eval()

data = StubDataset(dir="/Users/ritchie/Downloads/Smart Primer User Testing Location Photos")

classes = ["background","eucalyptus","tree"]

for i in range(len(data)):
    data_item = data[i]
    image = data_item["image"]
    filename = data_item["filename"]
    print(filename + "," + classes[model(image.resize(1,3,224,224)).argmax()])


