import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from backbones import get_model
import argparse
from datetime import datetime
import imutils
import sys
from sklearn.svm import SVC
import dlib
from sklearn.cluster import KMeans

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = get_model('r50', fp16=False)
model.load_state_dict(torch.load('/app/Recognition/arcface_torch/backbone.pth'))
model.eval()
print("modelo carregado!")

predictor_path = "./shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

def crop_images(image_file):

    face_file_path = image_file
    print(face_file_path)
    img = dlib.load_rgb_image(face_file_path)

    dets = detector(img, 1)
    crop_found = []
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
    else:
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
        #images = dlib.get_face_chips(img, faces, size=320)
        for i,image in enumerate(images):
            #im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            im_pil = Image.fromarray(image)
            crop_found.append(im_pil)
    return crop_found

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC


def main():
    train_dir = '/app/Recognition/arcface_torch/pessoas'
    print('Diretório de treino carregado!')
    features = []
    labels = []

    centroids_by_class = {}

    for character in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, character)):
            continue
        print(f'Diretório de classe {character} carregado!')
        
        features_by_class = []
        for img_file in os.listdir(os.path.join(train_dir, character)):
            img_path = os.path.join(train_dir, character, img_file)
            img = cv2.imread(img_path)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 64))
            binarie = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            binarie[binarie == 255] = 1
            hog = cv2.HOGDescriptor((128, 64), (8, 8), (4, 4), (8, 8), 9)
            hog_feat = hog.compute(np.uint8(binarie)).flatten()
            
            features_by_class.append(hog_feat)

        features_by_class = np.array(features_by_class)
        kmeans = KMeans(n_clusters=8)
        kmeans.fit(features_by_class)
        centroids = kmeans.cluster_centers_
        centroids_by_class[character] = centroids

        features.extend(centroids)
        labels.extend([character] * len(centroids))

    clf = SVC()
    clf.fit(features, labels)

    img_path = '/app/Recognition/arcface_torch/testes/cda.jpg'
    crops = crop_images(img_path)
    print(len(crops))
    predicted_class = []
    for crop in crops:
        crop_np = np.asarray(crop)
        gray = cv2.cvtColor(crop_np, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 64))
        binarie = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        binarie[binarie == 255] = 1
        hog = cv2.HOGDescriptor((128, 64), (8, 8), (4, 4), (8, 8), 9)
        hog_feat = hog.compute(np.uint8(binarie)).flatten()
        
        distances = []
        for character, centroids in centroids_by_class.items():
            dist = np.linalg.norm(centroids - hog_feat, axis=1)
            distances.append((character, np.sum(dist)))
        
        predicted_class.append(min(distances, key=lambda x: x[1])[0])

        print(predicted_class)

    return predicted_class


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(
    #    description="Distributed Arcface Training in Pytorch")
    #parser.add_argument("path", type=str, help="the path to the image")
    main()