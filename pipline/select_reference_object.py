import os
import cv2
import numpy as np
import shutil

def copy_and_rename_images(filename):
    src_dir = '../XMem2/workspace/demo/segments'
    dst_dir = './demo'
    # filename = 'frame_000002.jpg'
    new_filename = '1.jpg'
    bg_filename = '1_bg.png'

    # Copy the image to the specific directory and rename it
    shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, new_filename))

    # Find the corresponding image and copy it to the specific directory
    mask_dir = os.path.join(os.path.dirname(src_dir), 'masks')
    mask_filename = os.path.splitext(filename)[0] + '.png'
    mask_filename = os.path.basename(mask_filename)
    # print(os.path.splitext(filename))
    # print(mask_filename)
    # print(mask_dir)
    # print(os.path.join(mask_dir, mask_filename))
    shutil.copy(os.path.join(mask_dir, mask_filename), os.path.join(dst_dir, bg_filename))

def display_images(directory_path, callback=None):
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

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            col = x // width
            row = y // height
            index = row * cols + col
            if index < len(images):
                if callback is not None:
                    # callback(os.path.join(directory_path, os.listdir(directory_path)[index]))
                    callback(os.listdir(directory_path)[index])
                cv2.destroyAllWindows()

    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Images', on_click)
    cv2.imshow('Images', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def callback(file_path):
    print(file_path)
    copy_and_rename_images(file_path)
    
display_images('../XMem2/workspace/demo/segments/', callback=callback)