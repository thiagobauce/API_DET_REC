"""
Here will be uptade to measure the metrics with just embeddings without knn.
"""

import os
import torch
from backbones import get_model
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from utils.utils_config import get_config
import numpy as np
import torch
from backbones import get_model
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 1
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12585",
        rank=rank,
        world_size=world_size,
    )

transform = transforms.Compose([transforms.Resize((130,130)),
                                transforms.CenterCrop((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])

cfg = get_config('configs/report')    

model = get_model('r50', fp16=False)
#model.fc = nn.Linear(model.fc.in_features,512)
#model.classifier = nn.Linear(512,19)
model.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/model.pt', map_location='cpu'))

# Carregar o conjunto de teste
test_ds = torchvision.datasets.ImageFolder("/app/Recognition/arcface_torch/pessoas", transform=transform)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32)

# Listas para armazenar os rótulos reais e as predições
true_labels = []
pred_labels = []

# Fazer previsões no conjunto de teste
for images, labels in test_dl:
    with torch.no_grad():
        images = images.to('cpu')
        labels = labels.to('cpu')
        embeddings = model(images)
        predictions = torch.argmax(embeddings, dim=1)
    
    true_labels.extend(labels.cpu().numpy().tolist())
    pred_labels.extend(predictions.cpu().numpy().tolist())

# Calcular as métricas usando os rótulos reais (true_labels) e as predições (pred_labels)
precision = metrics.precision_score(true_labels, pred_labels, average='macro')
recall = metrics.recall_score(true_labels, pred_labels, average='macro')
f1 = metrics.f1_score(true_labels, pred_labels, average='macro')

print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)