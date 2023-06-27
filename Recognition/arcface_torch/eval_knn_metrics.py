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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.preprocessing import LabelEncoder

model = get_model('r50', fp16=False)
model.fc = nn.Linear(model.fc.in_features,512)
model.classifier = nn.Linear(512,19)
model.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/model_finnedfc_234.pt',map_location=torch.device('cpu')))

#model = torchvision.models.resnet50(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features,512)
#model.classifier = nn.Linear(model.fc.in_features,512)
#model.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/last_run.pth',map_location=torch.device('cpu'))['model'])

transform = transforms.Compose([transforms.Resize((130,130)),
                                transforms.CenterCrop((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])

ds = torchvision.datasets.ImageFolder("/app/Recognition/arcface_torch/megaface/data/facescrub_images",transform=transform)

nomes = np.asarray(ds.classes)

dl = torch.utils.data.DataLoader(ds,batch_size=256)

model.eval()

x,y = next(iter(dl))

predi = model(x)

xall = np.asarray(predi.tolist())

xtrain, xtest, ytrain, ytest = train_test_split(xall, y, test_size=0.33, random_state=42,stratify=y)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,weights='distance')

knn.fit(xtrain,ytrain)

pred = knn.predict(xtest)

#print(metrics.classification_report(ytest,pred))

accuracy = accuracy_score(ytest, pred)
precision = precision_score(ytest, pred,average='macro')
recall = recall_score(ytest, pred,average='macro')
f1 = f1_score(ytest, pred, average='macro')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
