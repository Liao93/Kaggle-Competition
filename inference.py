import os
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

img_size = 285

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.regnet_y_8gf(pretrained=False) # models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)
model = model.to(device)
model.load_state_dict(torch.load("model_fish.pkl"))

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model.eval()
files = os.listdir('./data/test_stg1')
softm = nn.Softmax(dim=1)
filenames = []
preds = []
with torch.no_grad():
    # image order is important to your result
    for item in tqdm(files):
        if os.path.isfile(os.path.join('./data/test_stg1', item)):
            img = Image.open(os.path.join('./data/test_stg1', item))
            img = img.convert('RGB')
            img = transform(img)
            img = img.unsqueeze_(0).to(device)
            output = model(img)
            output = softm(output)
            output = output[0].cpu()
            o = [x.item() for x in output]
            filenames.append(item)
            preds.append(o)

files = os.listdir('./data/test_stg2')
with torch.no_grad():
    # image order is important to your result
    for item in tqdm(files):
        if os.path.isfile(os.path.join('./data/test_stg2', item)):
            img = Image.open(os.path.join('./data/test_stg2', item))
            img = img.convert('RGB')
            img = transform(img)
            img = img.unsqueeze_(0).to(device)
            output = model(img)
            output = softm(output)
            output = output[0].cpu()
            o = [x.item() for x in output]
            filenames.append('test_stg2/' + item)
            preds.append(o)

preds = np.array(preds).T 
dic = {
    "image": filenames, 
    "ALB": preds[0],
    "BET": preds[1],
    "DOL": preds[2],
    "LAG": preds[3],
    "NoF": preds[4],
    "OTHER": preds[5],
    "SHARK": preds[6],
    "YFT": preds[7],    
}
df = pd.DataFrame(dic)
df.to_csv("predict.csv", index=False)