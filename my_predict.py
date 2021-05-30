import os
import time
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

from pspnet import Pspnet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    start = time.time()
    psp_net = Pspnet()

    img_base_path = r'D:\study\大三下\cv\21\作业\cv21b.programming03\dataset\images'
    txt_path = r'D:\study\大三下\cv\21\作业\cv21b.programming03\train.txt'
    imgs = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            imgs.append(line.strip())
    for img in imgs:
        try:
            image = Image.open(os.path.join(img_base_path, img + '.jpg'))
        except:
            print("Open error: " + img)
            continue
        else:
            seg_img = psp_net.detect_image(image)
            seg_img.save(os.path.join(r'./datasets/train', img + '.png'))
    end = time.time()
    print(end - start)
