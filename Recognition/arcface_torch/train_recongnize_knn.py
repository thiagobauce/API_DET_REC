import cv2
import imutils
import numpy as np
import os
import sys
from PIL import Image,ImageStat
import argparse
import torch
from backbones import get_model
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sklearn.neighbors
import sklearn.metrics as metrics
import joblib
from sklearn.preprocessing import LabelEncoder

model = get_model('r50', fp16=False)
model.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/model.pt',map_location=torch.device('cpu')))

#model = torchvision.models.resnet50(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features,512)
#model.classifier = nn.Linear(model.fc.in_features,512)
#model.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/last_run.pth',map_location=torch.device('cpu'))['model'])

transform = transforms.Compose([transforms.Resize((130,130)),
                                transforms.CenterCrop((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])

ds = torchvision.datasets.ImageFolder("/app/Recognition/arcface_torch/pessoas",transform=transform)

nomes = np.asarray(ds.classes)

dl = torch.utils.data.DataLoader(ds,batch_size=len(ds))

model.eval()

x,y = next(iter(dl))

pred = model(x)

xall = np.asarray(pred.tolist())

xtrain, xtest, ytrain, ytest = train_test_split(xall, y, test_size=0.33, random_state=42,stratify=y)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance')

knn.fit(xtrain,ytrain)

joblib.dump(knn, 'knn_trained.pkl')

pred = knn.predict(xtest)

print(nomes)

print(metrics.classification_report(ytest,pred))

print(metrics.confusion_matrix(ytest,pred))
