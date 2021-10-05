import cv2
import matplotlib.pyplot as plt


def read_img(img_path):
    # Reads an image from disc
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError('Image was not read!', img_path)
    return img

def show_img(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,5))
    plt.imshow(img)
    plt.show()
    