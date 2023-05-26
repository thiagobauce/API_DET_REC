class FacesEmoreDataset(Dataset):
    def __init__(self, data_path):
        # Carrega o conjunto de dados Faces Emore a partir do caminho especificado
        self.data = torch.load(data_path)

    def __getitem__(self, index):
        # Retorna uma imagem de entrada, um rótulo de classe e um rótulo de identidade
        img, class_id, identity_id = self.data[index]
        return img, class_id, identity_id

    def __len__(self):
        # Retorna o número de itens no conjunto de dados
        return len(self.data)

train_dataset = FacesEmoreDataset('path/to/train_faces_emore.pt')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)