import os
import cv2
import numpy as np

def display_images(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(directory_path, filename))
            images.append(img)

    rows = int(np.ceil(len(images) / 2))
    cols = 2
    height, width, _ = images[0].shape
    canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        canvas[y:y+height, x:x+width, :] = img

    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    cv2.imshow('Images', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

display_images('./key_frames/')