import os
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import random

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set random seed for reproducibility
same_seeds(43)


img_size = 285
argument_num = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.regnet_y_8gf(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)
model = model.to(device)
model.load_state_dict(torch.load("model_fish.pkl"))

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.Resize((img_size, img_size)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), 
                            shear=10, 
                            interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_no_aug = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model.eval()
files = os.listdir('./data/test_stg1')
softm = nn.Softmax(dim=1)
filenames = []
#preds = []
arg_preds = []
with torch.no_grad():
    # image order is important to your result
    for item in tqdm(files):
        if os.path.isfile(os.path.join('./data/test_stg1', item)):
            arg_output = torch.zeros(8)
            for n in range(argument_num):
                img = Image.open(os.path.join('./data/test_stg1', item))
                img = img.convert('RGB')
                if n == 0:
                    img = transform_no_aug(img)
                else:
                    img = transform(img)
                img = img.unsqueeze_(0).to(device)
                output = model(img)
                output = softm(output)
                output = output[0].cpu()
                arg_output += output
                #o = [x.item() for x in output]
                #preds.append(o)
            arg_output /= argument_num
            arg_o = [x.item() for x in arg_output]
            arg_preds.append(arg_o)
            filenames.append(item)

files = os.listdir('./data/test_stg2')
with torch.no_grad():
    # image order is important to your result
    for item in tqdm(files):
        if os.path.isfile(os.path.join('./data/test_stg2', item)):
            arg_output = torch.zeros(8)
            for n in range(argument_num):
                img = Image.open(os.path.join('./data/test_stg2', item))
                img = img.convert('RGB')
                if n == 0:
                    img = transform_no_aug(img)
                else:
                    img = transform(img)
                img = img.unsqueeze_(0).to(device)
                output = model(img)
                output = softm(output)
                output = output[0].cpu()
                arg_output += output
            arg_output /= argument_num
            arg_o = [x.item() for x in arg_output]
            arg_preds.append(arg_o)
            filenames.append('test_stg2/' + item)

arg_preds = np.array(arg_preds).T 
dic = {
    "image": filenames, 
    "ALB": arg_preds[0],
    "BET": arg_preds[1],
    "DOL": arg_preds[2],
    "LAG": arg_preds[3],
    "NoF": arg_preds[4],
    "OTHER": arg_preds[5],
    "SHARK": arg_preds[6],
    "YFT": arg_preds[7],    
}
df = pd.DataFrame(dic)
df.to_csv("predict.csv", index=False)