import sys
import os

package_path = os.path.dirname(os.path.abspath(__file__))
arcface_torch_path = os.path.abspath(os.path.join(package_path, '..'))
sys.path.insert(0, arcface_torch_path)

import itertools
import torch
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
from backbones import get_model
import time

transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device = torch.device('cuda:1')

if __name__ == '__main__':
    # Diretório raiz que contém os diretórios das identidades
    root_directory = "/app/Recognition/arcface_torch/pepo_ds/test"

    # Lista de identidades (pessoas) baseada nos diretórios presentes no diretório raiz
    identities = os.listdir(root_directory)

    # Dicionário de imagens por identidade
    images_per_identity = {}
    for identity in identities:
        identity_directory = os.path.join(root_directory, identity)
        image_files = os.listdir(identity_directory)
        image_paths = [os.path.join(identity_directory, image_file) for image_file in image_files]
        images_per_identity[identity] = image_paths

    # Criar pares positivos (mesma pessoa)
    positive_pairs = []
    for identity in identities:
        images = images_per_identity[identity]
        pairs = list(itertools.combinations(images, 2))
        positive_pairs.extend(pairs)
    print('positive: ',len(positive_pairs))

    # Criar pares negativos (pessoas diferentes)
    negative_pairs = []
    for pair in itertools.combinations(identities, 2):
        identity1, identity2 = pair
        images1 = images_per_identity[identity1]
        images2 = images_per_identity[identity2]
        pairs = list(itertools.product(images1, images2))
        negative_pairs.extend(pairs)
    print('negative: ', len(negative_pairs))

    start_time = time.time()

    #model_path = '/app/Recognition/arcface_torch/checkpoints/model_fi.pt'
    model_path = '/app/Recognition/arcface_torch/checkpoints/model_ruim.pt'
    model = get_model('r50', fp16=False)
    #model.fc = torch.nn.Linear(model.fc.in_features,512)
    #model.classifier = torch.nn.Linear(512,19)
    model.load_state_dict(torch.load(model_path, map_location='cuda:1'))
    model.to(device)
    model.eval()
    
    predict_pos = []
    for anchor, comparison in positive_pairs:
        # Carregar as imagens de anchor e comparison
        anchor_img = Image.open(anchor).convert('RGB')
        comparison_img = Image.open(comparison).convert('RGB')

        # Pré-processar as imagens
        anchor_img = transform(anchor_img).unsqueeze(0).to(device)
        comparison_img = transform(comparison_img).unsqueeze(0).to(device)

        # Extrair os embeddings das imagens
        with torch.no_grad():
            anchor_embedding = model(anchor_img)
            comparison_embedding = model(comparison_img)

        # Calcular a distância euclidiana entre os embeddings para realizar a verificação
        distance = cosine_similarity(anchor_embedding, comparison_embedding)
        if distance > 0.3:
            predict_pos.append(1)
        else:
            predict_pos.append(0)

    acc_pos = sum(predict_pos) / len(positive_pairs)

    print(sum(predict_pos))

    predict_neg = []
    for anchor, comparison in negative_pairs:
        # Carregar as imagens de anchor e comparison
        anchor_img = Image.open(anchor).convert('RGB')
        comparison_img = Image.open(comparison).convert('RGB')

        # Pré-processar as imagens
        anchor_img = transform(anchor_img).unsqueeze(0).to(device)
        comparison_img = transform(comparison_img).unsqueeze(0).to(device)

        # Extrair os embeddings das imagens
        with torch.no_grad():
            anchor_embedding = model(anchor_img)
            comparison_embedding = model(comparison_img)

        # Calcular a distância euclidiana entre os embeddings para realizar a verificação
        distance = cosine_similarity(anchor_embedding, comparison_embedding)
        if distance < 0.3:
            predict_neg.append(1)
        else:
            predict_neg.append(0)

    acc_neg = sum(predict_neg) / len(negative_pairs)

    print(sum(predict_neg))

    acc_media =  (acc_pos + acc_neg) / 2
    tot_imges = len(negative_pairs) + len(positive_pairs)

    end_time = time.time()

    print("Métricas:")
    print("Tempo de Inferência: ", (end_time - start_time))
    print("Quantidade de Imagens: ", tot_imges)
    print("Acurácia Média:", acc_media)
    print("Acurácia dos Positivos:", acc_pos)
    print("Acurácia dos Negativos:", acc_neg)
