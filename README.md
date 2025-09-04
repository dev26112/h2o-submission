# h2o-submission


## COCO Sky Removal Dataset Processor
This project implements a script to process a COCO-format dataset by automatically removing the sky region from images and adjusting the corresponding annotations. It utilizes deep learning models from Detectron2 to detect the skyline and performs the necessary data manipulation.

## Description
This script takes an existing COCO dataset (specifically tested with val2017) and creates a new dataset where images are cropped from below the detected skyline downwards. The bounding box and segmentation annotations in the COCO JSON file are adjusted to match the new cropped image dimensions. This allows for dataset manipulation without requiring re-annotation.

## Features
* Downloads a sample of the COCO val2017 dataset.
* Uses Detectron2's Panoptic Segmentation model (COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml) to detect the 'sky' class and determine the skyline.
* Includes a fallback mechanism using object detection heuristics if the sky class is not found by the panoptic model.
* Crops images below the detected skyline.
* Adjusts bounding box and polygon segmentation annotations for the cropped images.
* Generates a new COCO-format JSON annotation file for the cropped dataset.
* Provides visualization functions to validate the results by comparing original and cropped images with their respective annotations.

## Installation
These instructions will help to get a copy of the project up and running. This project is designed to run in a Python environment, preferably Google Colab or a local machine with a GPU.

## Prerequisites
* Python 3.x
* pip
* A machine capable of running Detectron2 (preferably with a GPU)
## Steps
* Set up the Environment: Run the environment setup commands. This typically involves installing PyTorch, Detectron2, pycocotools, and OpenCV.

```
# Example setup commands (adapted for potential Colab environment)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install pycocotools
pip install opencv-python
```

* Download Required Data: The script will automatically download a sample of the COCO val2017 images and annotations.
* Run the Script: Execute the main script cells. This will process the images, generate the cropped dataset and the new annotation file (annotations/instances_val2017_cropped.json), and display visualizations.

## Usage
1. Ensure the environment is set up and dependencies are installed.
2. Execute the script cells in order within your chosen environment (e.g., Jupyter Notebook, Google Colab).
3. The script will process a small sample of images by default. You can modify the sample_img_ids selection to process more or fewer images.
4. The output will be:
* Cropped images saved in the cropped_val2017 directory.
* A new COCO annotation file: annotations/instances_val2017_cropped.json.
* Visualizations comparing original and cropped images with annotations.

## Implementation Details
* Skyline Detection Model: COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml from Detectron2's model zoo.
* Fallback Detection Logic: If the panoptic model fails to identify the 'sky' class (ID 180), the script falls back to a heuristic based on the general object detection model's output, estimating the skyline based on the first row containing detected objects or defaulting to the top third of the image.
* Annotation Adjustment: Bounding boxes are adjusted by subtracting the crop offset. Polygon segmentations are adjusted by subtracting the offset from the Y-coordinates of the vertices, discarding polygons that fall entirely outside the new image bounds. RLE segmentation masks are currently not adjusted.
* Dataset Format: The output follows the standard COCO dataset format for instance segmentation/detection.

* Images (val2017.zip): This  contains 5,000 images. These are the actual pictures on which the computer vision model will identify and segment objects.

* Annotations (annotations_trainval2017.zip): This zip file contains the ground truth data for the images. The script uses the instances_val2017.json file from this archive. This JSON file includes:

* Bounding boxes: The coordinates for a box drawn around each object in an image.

* Segmentation masks: A pixel-by-pixel outline of each object, allowing for more precise shape identification.

* Category labels: Each object is assigned a category from a list of 80 "thing" categories (like "person," "car," or "dog") and 91 "stuff" categories (amorphous things like "sky" or "grass").

## How the Script Uses the Dataset
* Downloads Data: The first part of the script uses wget to download both the image zip file (val2017.zip) and the annotation zip file (annotations_trainval2017.zip) directly from the COCO dataset's web servers.

* Identifies the Sky: It processes these images and uses a panoptic segmentation model to specifically find the "sky" category.

* Creates a New, Cropped Dataset: After removing the sky, it generates a new set of images (cropped_val2017) and a new annotation file (instances_val2017_cropped.json) with adjusted coordinates for the bounding boxes and segmentation masks to match the smaller, cropped images.

* The script takes a standard, well-known computer vision dataset and modifies it by removing the sky to create a new, derivative dataset for potentially more focused model training.
