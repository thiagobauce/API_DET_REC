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
from losses import CombinedMarginLoss, ArcFace, CosFace
from partial_fc_v2 import PartialFC_V2, DistCrossEntropyFunc, DistCrossEntropy, AllGatherFunc

PYTORCH_CUDA_ALLOC_CONF = 0

if __name__ == '__main__':
    n_classes = 10177
    emb_size = 512
    batch_size = 128
    initial_lr = 0.001
    lr = initial_lr

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.Resize((50,50)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])

    ds_train = torchvision.datasets.CelebA('./', split='train', target_type='identity', transform=transform, download=True)
    ds_test = torchvision.datasets.CelebA('./', split='test', target_type='identity', transform=transform, download=True)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, drop_last=True)
    dl_test  = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, drop_last=True)

    x, y = next(iter(dl_train))
    print('aqui')

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, emb_size)
    model.classifier = nn.Linear(model.fc.in_features, emb_size)

    model.to(device)

    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12585",
        rank=rank,
        world_size=world_size,
    )

    print('foi')

    margin_loss = ArcFace()
    loss_function = PartialFC_V2(margin_loss=margin_loss, embedding_size=emb_size, num_classes=n_classes, sample_rate=0.4)
    loss_function.train().to(device)

    opt = optim.AdamW(model.parameters(), lr=lr)
    stop = False

    epoch = 0
    patience = 10
    start_time = time.perf_counter()
    batch_loss = []
    batch_loss_test = []
    best_loss = 10000
    stable_loss_counter = 0  # Contador para verificar estabilidade da perda de teste
    stable_loss_threshold = 1e-3  # Limiar de diferença de perda para considerar como estável
    print('foi')
    while not stop:
        batch_list_loss = []
        batch_iterator = tqdm(dl_train)

        for i, (x, y) in enumerate(batch_iterator):
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x)
            loss = loss_function(embeddings, y)

            opt.zero_grad()
            loss.backward()
            batch_list_loss.append(loss.item())
            opt.step()

        batch_loss.append(np.mean(batch_list_loss))
        print("training loss", batch_loss[-1])

        with torch.no_grad():
            batch_list_loss = []
            for i, (x, y) in enumerate(dl_test):
                x = x.to(device)
                y = y.to(device)
                embeddings = model(x)
                loss = loss_function(embeddings, y)
                batch_list_loss.append(loss.item())
            batch_loss_test.append(np.mean(batch_list_loss))
        print("test loss", batch_loss_test[-1])

        if batch_loss_test[-1] < best_loss:
            print("saving model")
            patience_wait = patience
            best_loss = batch_loss_test[-1]
            save_model = {'model': model.state_dict(), 'opt': opt.state_dict(), 'loss_training': batch_loss, 'epoch': epoch}
            torch.save(save_model, f"best_model_v1_{epoch}.pth")
        else:
            stable_loss_counter += 1
            if stable_loss_counter >= patience:
                print("Test loss has stabilized. Stopping training.")
                stop = True

        if batch_loss[-1] < 1.5:
            patience_wait -= 1
            if patience_wait == 0:
                print("Training loss has stabilized. Stopping training.")
                stop = True

        print("saving model")
        save_model = {'model': model.state_dict(), 'opt': opt.state_dict(), 'loss_training': batch_loss, 'epoch': epoch}
        torch.save(save_model, 'best_model_{}.pth'.format(epoch))
        print('Epoch: {} || LR: {}  ||  Patience_wait: {} '.format(epoch, lr, patience_wait))
        epoch += 1

    end_time = time.perf_counter()

    save_model = {'model': model.state_dict(), 'opt': opt.state_dict(), 'loss_training': batch_list_loss, 'epoch': epoch}
    torch.save(save_model, 'last_run.pth')
