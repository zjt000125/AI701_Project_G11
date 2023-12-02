<<<<<<< HEAD
# Fast and Interactive Text-to-Image Video Object Segmentation, Editing and Derivative Creation

## Abstract

Current mainstream deep learning methods supporting user-defined image generation require single-centered-object images as input. Moreover, these methods often necessitate extensive training periods before they can effectively generate new images. In this work, we present a novel pipeline capable of handling video inputs containing multiple interfering foreground objects. Our approach allows users to specify the segmentation target in multi-object videos. We trained a domain adaptor and an image transformer module to extract the high-level semantic information and low-level texture information, which are then seamlessly integrated with state-of-the-art lightweight text-to-image method stable diffusion through fine-tune the cross attention part in U-Net. In less than 10 seconds, our model can learn the new image and connect it with a pseudo word 'S*', and generate a new 512X512 images with background replacement or editing based on textual prompts, providing a rapid and efficient solution for video object extracting and editing.

## Getting Started

### Environment Setup

```shell
git clone git@github.com:zjt000125/AI701_Project_G11.git
cd AI701_Project_G11
conda env create -f ai701.yml
conda env create -f xmem2.yml
cd ..
git clone git@github.com:max810/XMem2.git
sudo apt update
sudo apt install ffmpeg
```

The download link for the checkpoints, dataset and scrips are here: 
https://mbzuaiac-my.sharepoint.com/personal/jiantong_zhao_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjiantong%5Fzhao%5Fmbzuai%5Fac%5Fae%2FDocuments%2FAI701%5FProject&ga=1

Download mask_process.py and Images2Segments.py, and put them into **./XMem2/util/** directory.

### Pretrained Model

download the pretrained model and put the checkpoints in **'./Ai701_Project_G11/checkpoints/'** directory.

### Fast Generation Using Your Images

#### Directly Generate from Segmented Image:

```sh
bash generate.sh
```

#### The Whole Pipeline

```shell
bash demo.sh

bash generate.sh
```

The display demo video is:
<video src="AI701_Project_G11/demo_show_video/demo.mp4"></video>

* extract key frames from the ./videos/three_cats.mp4 through ffmpeg
* display the key frames on the screen(Press any key to continue)
* launch the XMem2 interactive segmentation demo to segment the key frames for reference object(Click on the first frame or draw a circle to select the object to be segmented, and click save references, and click full propagate to do the segmentation for other key frames)
* get the mask and segmented object image saved in ./demo/

## Train the models

Download the datasets from this link:

unzip the datasets and put it directly into the **Ai701_Project_G11/** directory      
The final directory structure is: 
**Ai701_Project_G11/datasets**
### Train Domain Adaptor

```shell
bash train_domain_adaptor.sh
```

### Train Image Transformer

```shell
bash train_image_transformer.sh
```

=======
# AI701_Project_G11
>>>>>>> 530002d66dc272d6f545d4fe14a3ea4d66a2f5b7
