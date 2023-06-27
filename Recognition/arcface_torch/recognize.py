import os
import argparse
from datetime import datetime
from backbones import get_model
import dlib
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image,ImageStat
import PIL.Image as Image
import imutils
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neighbors
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
        transforms.Resize((130, 130)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

model = get_model('r50', fp16=False)
#model.fc = nn.Linear(model.fc.in_features,512)
#model.classifier = nn.Linear(512,19)
model.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/model.pt',map_location=torch.device('cpu')))
model.eval()

ds = torchvision.datasets.ImageFolder('/app/Recognition/arcface_torch/pessoas', transform=transform)
dl = torch.utils.data.DataLoader(ds,batch_size=len(ds))
nomes = np.asarray(ds.classes)

x,y = next(iter(dl))
pred = model(x)
xall = np.asarray(pred.tolist())
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,weights='distance')
knn.fit(xall,y)

def get_person_ids(crops):
    imglist = []
    for img in crops:
        im_pil = Image.fromarray(img)
        imglist.append(transform(im_pil).unsqueeze(0))
    x = torch.cat(imglist,axis=0)
    embeddings = model(x)
    embeddings = np.asarray(embeddings.tolist())
    prob_pessoas = knn.predict_proba(embeddings)
    id_pessoas = prob_pessoas.argmax(axis=1)
    #id_pessoas = []
    #print(len(embeddings))
    #for embedding in embeddings:
    #    prob_pessoa = knn.predict_proba(embedding.reshape(1, -1))[0]
    #    id_pessoa = np.argmax(prob_pessoa)
    #    id_pessoas.append(id_pessoa)
    #    print(nomes[id_pessoas])
    return nomes[id_pessoas]

def encontra_pessoas(image_file):
    os.chdir('/app/Detection')
    print(image_file)
    os.system(f'python detect.py {image_file}')

    now = datetime.now()
    date_str = now.strftime("%Y/%m/%d")
    crop_dir = os.path.join('/app/Recog-API', date_str, 'cropped_images')
    crops = [cv2.imread(os.path.join(crop_dir, filename)) for filename in os.listdir(crop_dir)]
    #print(len(crops))
    ids = get_person_ids(crops)
    print(ids)

    for crop, id, filename in zip(crops, ids, os.listdir(crop_dir)):
        extension = os.path.splitext(filename)[1]
        new_filename = str(id) + extension
        os.rename(os.path.join(crop_dir, filename), os.path.join(crop_dir, new_filename))
    
    return ids


def main(args):
    image_file = args.path
    people = encontra_pessoas(image_file)
    return people


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("path", type=str, help="the path to the image")
    main(parser.parse_args())
