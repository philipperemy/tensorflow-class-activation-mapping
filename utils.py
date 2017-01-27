from glob import glob

import numpy as np
import skimage.io
import skimage.transform


def read_dataset(percentage=1.0):
    labels = []
    with open('data/labels.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            [_, label] = l.strip().split('\t')
            labels.append(int(label))
    labels = np.array(labels)
    images = []
    N = len(glob('data/img/*.png'))
    assert N == len(labels)
    max_images = int(percentage * N)
    labels = labels[:max_images]
    print('found {} images.'.format(N))
    for i in range(1, N + 1):
        f = 'data/img/img_{}.png'.format(i)
        images.append(load_image(f))
        if i % 1000 == 0:
            print('read {} images.'.format(i))
        if len(images) == max_images:
            break
    images = np.array(images)
    print(images.shape)
    assert max_images == len(images)
    assert max_images == len(labels)
    return images, labels


def load_image(path):
    try:
        img = skimage.io.imread(path).astype(float)
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img = np.tile(img[:, :, None], 3)
    if img.shape[2] == 4: img = img[:, :, :3]
    if img.shape[2] > 4: return None

    img /= 255.

    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resized_img = skimage.transform.resize(crop_img, [224, 224])
    return img


if __name__ == '__main__':
    read_dataset(0.1)
