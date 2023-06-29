import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import os
from backbones import get_model
from partial_fc_v2 import PartialFC_V2
from losses import CombinedMarginLoss
from utils.utils_config import get_config
import argparse

    #torch.cuda.get_device_properties(0)

def main(args):
    n_classes = 19
    emb_size = 512
    lr = 0.01
    batch_size = 16

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """# Prepare data loaders"""

    cfg = get_config(args.config)

    transform = transforms.Compose([transforms.Resize((112,112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])

    ds_train = torchvision.datasets.ImageFolder('/app/Recognition/arcface_torch/pepo_ds/train',train=True,transform=transform)
    ds_test  = torchvision.datasets.ImageFolder('/app/Recognition/arcface_torch/pepo_ds/test',train=False,transform=transform)

    n_classes = len(ds_train.classes)

    dl_train = torch.utils.data.DataLoader(ds_train,batch_size=batch_size,drop_last=True)
    dl_test  = torch.utils.data.DataLoader(ds_test,batch_size=batch_size,drop_last=True)

    x,y = next(iter(dl_train))

    """# Prepare model"""

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone.train()
    backbone.load_state_dict(torch.load('/app/Recognition/arcface_torch/checkpoints/model.pt'))

    #Fine-Tuning com ou sem congelamentos
    
    #for param in backbone.layer1.parameters():
    #    param.requires_grad = False
    #for param in backbone.layer2.parameters():
    #    param.requires_grad = False
    #for param in backbone.layer3.parameters():
    #    param.requires_grad = True
    #for param in backbone.layer4.parameters():
    #    param.requires_grad = False
    #backbone.fc = torch.nn.Linear(backbone.fc.in_features,512)
    #backbone.classifier = torch.nn.Linear(512,19)

    backbone = backbone.cuda()
    margin_loss = CombinedMarginLoss(
        64,
        1.00,
        0.5,
        0.0,
        0
    )

    loss_function = PartialFC_V2(margin_loss=margin_loss,embedding_size=emb_size, num_classes=n_classes, sample_rate=0.3)
    loss_function.train().cuda()
    """# Training """

    opt       = optim.SGD(
                params=[{"params": backbone.parameters()}, {"params": loss_function.parameters()}],
                lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    stop = False

    epoch = 0
    patience = 10
    start_time = time.perf_counter()
    batch_loss = []
    batch_loss_test = []
    best_loss = 10000

    start_time = time.perf_counter()
    patience = 10
    stop = False
    while(not stop):
        batch_list_loss = []
        batch_iterator = tqdm(dl_train)
        for i,(x,y)  in enumerate(batch_iterator):
            x = x.to(device)
            y = y.to(device)
            embeddings = backbone(x)
            loss = loss_function(embeddings,y)

            opt.zero_grad()
            loss.backward()
            batch_list_loss.append(loss.item())
            opt.step()
        batch_loss.append(np.mean(batch_list_loss))
        print("training loss ",batch_loss[-1])

        with torch.no_grad():
            batch_list_loss = []
            for i,(x,y) in enumerate(dl_test):
                x = x.to(device)
                y = y.to(device)
                embeddings = backbone(x)
                loss = loss_function(embeddings,y)
                batch_list_loss.append(loss.item())
            batch_loss_test.append(np.mean(batch_list_loss))
        print("test loss ",batch_loss_test[-1])
        if batch_loss_test[-1] < best_loss:
            print("saving model")
            patience_wait = patience
            best_loss = batch_loss_test[-1]
            path_module = os.path.join(cfg.output, "model_ruim.pt")
            torch.save(backbone.state_dict(), path_module)
        patience_wait -= 1
        if patience_wait == 0:
            stop = True

    epoch +=1
    end_time = time.perf_counter()

    path_module = os.path.join(cfg.output, "model_ruim.pt")
    torch.save(backbone.state_dict(), path_module)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())