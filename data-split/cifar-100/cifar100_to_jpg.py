import os
import numpy as np
import pickle
import cv2

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

# Load the downloaded data
data_path = '/data/put_data/cclin/datasets/cifar-100-python/'
meta = unpickle(os.path.join(data_path, 'meta'))
train = unpickle(os.path.join(data_path, 'train'))
test = unpickle(os.path.join(data_path, 'test'))

# Save numpy arrays into png files
png_directory = os.path.join(data_path, 'images_png')
if not os.path.exists(png_directory):
    os.makedirs(png_directory)
for idx in range(len(train[b'data'])):
    img = train[b'data'][idx].reshape((3, 32, 32)).transpose([1, 2, 0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fine_label_name = meta[b'fine_label_names'][train[b'fine_labels'][idx]].decode('ascii')
    if not os.path.exists(os.path.join(png_directory, fine_label_name)):
        os.makedirs(os.path.join(png_directory, fine_label_name))
    cv2.imwrite(os.path.join(png_directory, fine_label_name, train[b'filenames'][idx].decode('ascii')), img)
for idx in range(len(test[b'data'])):
    img = test[b'data'][idx].reshape((3, 32, 32)).transpose([1, 2, 0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fine_label_name = meta[b'fine_label_names'][test[b'fine_labels'][idx]].decode('ascii')
    if not os.path.exists(os.path.join(png_directory, fine_label_name)):
        os.makedirs(os.path.join(png_directory, fine_label_name))
    cv2.imwrite(os.path.join(png_directory, fine_label_name, test[b'filenames'][idx].decode('ascii')), img)

# Convert png to jpg
from PIL import Image
jpg_directory = os.path.join(data_path, 'images')
if not os.path.exists(jpg_directory):
    os.makedirs(jpg_directory)
for fine_label_name in os.listdir(png_directory):
    if not os.path.exists(os.path.join(jpg_directory, fine_label_name)):
        os.makedirs(os.path.join(jpg_directory, fine_label_name))
    for item in os.listdir(os.path.join(png_directory, fine_label_name)):
        if '.png' in item:
            img = Image.open(os.path.join(png_directory, fine_label_name, item))
            img.save(os.path.join(jpg_directory, fine_label_name, item.replace('.png', '.jpg')))
