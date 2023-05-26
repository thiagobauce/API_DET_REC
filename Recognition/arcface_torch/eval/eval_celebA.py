import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pdb
import time
import torchsummary
import os
from torch import distributed
import math

emb_size = 512
batch_size = 128

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((50,50)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
ds_train = torchvision.datasets.CelebA('/app/Recognition/arcface_torch/', split = 'train',target_type='identity',transform=transform, download=True)
ds_test = torchvision.datasets.CelebA('/app/Recognition/arcface_torch/', split = 'test',target_type='identity',transform=transform, download=True)
dl_train = torch.utils.data.DataLoader(ds_train,batch_size=batch_size,drop_last=True)
dl_test  = torch.utils.data.DataLoader(ds_test,batch_size=batch_size,drop_last=True)

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features,emb_size)
model.classifier = nn.Linear(model.fc.in_features,emb_size)

# Carregar o modelo treinado
checkpoint = torch.load('/app/Recognition/arcface_torch/checkpoints/last_run.pth')  # Substitua pelo caminho do modelo treinado
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Avaliação do modelo
label_map = {}  # Dicionário para mapear rótulos únicos para números inteiros sequenciais
total_correct = 0
total_samples = 0

with torch.no_grad():
    for x, y in dl_test:
        x = x.to(device)
        y = y.to(device)
        embeddings = model(x)
        _, predicted_labels = torch.max(embeddings, 1)  # Obter as etiquetas preditas
        
        # Converter rótulos para números inteiros sequenciais
        for i in range(len(y)):
            label = y[i].item()
            if label not in label_map:
                label_map[label] = len(label_map)
            y[i] = label_map[label]
        
        total_correct += (predicted_labels == y).sum().item()
        total_samples += y.size(0)

accuracy = total_correct / total_samples
print("Accuracy on test set: {} / {}".format(total_correct,total_samples ))
