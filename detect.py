from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg_re50
from loss.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import argparse
import os
from datetime import datetime


cpu = True
confidence_threshold = 0.02
top_k = 100 #how many of the best face detections to keep after initial sorting of results.
nms_threshold = 0.4
keep_top_k = 50
save_image = True
visualization_threshold = 0.6


''' This function removes a common prefix from a collection of braces in a Python dictionary. 
    The main goal is to remove the 'module' prefix. 
    Which may be present in all parameter names of a model trained on an old version of PyTorch. '''
def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def show_image(faces,img_raw, image_path):
    for f in faces:
        if f[4] < visualization_threshold:
            continue
        text = "{:.4f}".format(f[4])
        f = list(map(int, f))
        cv2.rectangle(img_raw, (f[0], f[1]), (f[2], f[3]), (0, 0, 255), 2)
            
        cx = f[0]
        cy = f[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landmarks
        cv2.circle(img_raw, (f[5], f[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (f[7], f[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (f[9], f[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (f[11], f[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (f[13], f[14]), 1, (255, 0, 0), 4)

     # save image
    cv2.imwrite(f"{image_path}/result.jpg", img_raw)


def main(args):
    os.chdir('/app/Recog-API')
    image_path = args.path
    #print(image_path)
    torch.set_grad_enabled(False)
    cfg = cfg_re50
    model = RetinaFace(phase = 'test')
    model_path = '../Detection/checkpoints/Resnet50_final.pth'
    
    if cpu:
        saved_model = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        saved_model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in saved_model.keys():
        saved_model = remove_prefix(saved_model['state_dict'], 'module.')
    else:
        saved_model = remove_prefix(saved_model, 'module.')
    
    check_keys(model, saved_model)

    model.load_state_dict(saved_model, strict=False)
    model.eval()

    print('Finished loading model!')
    #print(model)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    model = model.to(device)

    resize = 1
    
    for i in range(100):
        now = datetime.now()
        date_str = now.strftime("%Y/%m/%d")

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = scale.to(device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        
        for_time = time.time()
        loc, conf, landmarks = model(img)
        print('net forward time: {:.4f}'.format(time.time() - for_time))

        prior_box = PriorBox(image_size=(im_height, im_width))
        anchors = prior_box.forward()
        anchors = anchors.to(device)
        prior_data = anchors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, cfg['variance'])
        scale2 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale2 = scale2.to(device)
        landmarks = landmarks * scale2 / resize
        landmarks = landmarks.cpu().numpy()

        lows = np.where(scores > confidence_threshold)[0]
        boxes = boxes[lows] #ignore lows conf boxes
        landmarks = landmarks[lows] #ignore lows lands
        scores = scores[lows] #ignore lows scores

        # keep top-K before Non-Maximum Suppression(NMS)
        order = scores.argsort()[::-1][:top_k] #order by best scores
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # NMS reduce redundancy in dets
        faces = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(faces, nms_threshold)
        faces = faces[keep, :]
        landmarks = landmarks[keep]

        # keep bests finals dets
        faces = faces[:keep_top_k, :]
        landmarks = landmarks[:keep_top_k, :]

        faces = np.concatenate((faces, landmarks), axis=1)
        
        os.chdir('/app/Recog-API/'+date_str)
        if not os.path.exists('cropped_images'):
            os.makedirs('cropped_images')
                
        # crop each bounding box in a separate image
        for i, det in enumerate(faces):
            if det[4] < visualization_threshold:
                continue
            x1, y1, x2, y2 = det[:4]
            cropped_img = img_raw[int(y1):int(y2), int(x1):int(x2), :]
            resized_image = cv2.resize(cropped_img, (150, 150))
            cv2.imwrite(f"cropped_images/cropped_image_{i}.jpg", resized_image)
            im_height, im_width, _ = resized_image.shape
            #print(f'altura: {im_height} -||- largura: {im_width}')
        # show image
        #show_image(faces,img_raw, image_path)
        
    return 'ok'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("path", type=str, help="the path to images")
    main(parser.parse_args())
    