import os
from PIL import Image

def resize_images(directory, target_size):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                image = Image.open(file_path)
                resized_image = image.resize(target_size, Image.ANTIALIAS)
                resized_image.save(file_path)
            except Exception as e:
                print(f"Erro ao redimensionar a imagem {file_path}: {str(e)}")

# Diretório raiz que contém as imagens
directory = '/app/Recognition/arcface_torch/pessoas'

# Tamanho alvo para redimensionamento
target_size = (112, 112)

# Redimensionar as imagens
resize_images(directory, target_size)
