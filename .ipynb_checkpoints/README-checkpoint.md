# Capstone: Mask Up
---

## Introduction 
The COVID-19 pandemic is an ongoing global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Currently [evidence](https://www.cdc.gov/coronavirus/2019-ncov/science/science-briefs/sars-cov-2-transmission.html) suggest that the transmission of COVID-19 is primary through respiratory droplets which can be expelled through coughs, sneezes, talks etc. As such, effective ways of inhibiting the spread of the virus is through wearing of surgical mask at public spaces. Mask wearing will reduce transmission from inhalation of virus and deposition of virus on exposed membranes.  

Countries around the world have made mask-wearing mandatory in public area as part of the measure to curb the the spread of COVID-19. In Singapore manpower resources are poured into to ensure compliance and enforcement of this rule. 

## Problem Statement 
Instead of deploying human resources to conduct mask-wearing compliance check or enforcement, they can be re-deployed to assist front line workers to manage the pandemic (e.g. health care workers etc). This project will explore using computer vision and deep learning for mask-wearing detection.

The problem to tackle falls under the subset of object detection problem in the machine vision domain. Specifically, this is a supervised classification problem where model is required to classify if a person is (1) wearing mask, (2) not wearing mask, or (3) not wearing mask correctly. 

## Downloading of Dataset
As this project is a supervised classification problem, sufficient label image data is required for training. Fortunately, a Kaggle user has shared this dataset online and can be downloaded through this [link](https://www.kaggle.com/andrewmvd/face-mask-detection). Otherwise, images of people wearing mask need to be scrape and annotated manually. 

It is important to note that annotated images served as the ground truth.  

The author has kindly provided 853 labelled image and annotation files. 

## Data Cleaning 
### Cleaning of Duplicated Images
Using difPy library, 5 duplicated images are identified. They are removed to prevent data leakage.

### Correction to dataset classes
Rename the target classes in the annotation files (*.xml):
- without_mask > no_mask
- with_mask > mask
- mask_weared_incorrect > incorrect


## EDA
### Inspect Quality of Images and Annotation Files
Some of images and annotation files are printed and check for their quality. As their served as the ground truth for our model's training, it is important that the quality is acceptable and consistent. 

Sample:

![ground_truth.png](assets/ground_truth.png)

### Check Class distribution
The class distribution of the downloaded images is inspected. 

```python
mask         0.791843
no_mask      0.178065
incorrect    0.030092
```

## Preprocessing

For preprocessing, Roboflow API to do the follwing preprocessing steps:

*1. Train, validation, test split* - 70%, 20%, 10%

*2. Image resizing* - 640*640 and 320*320 

*3. image augmentation* - horizontal flip, rotation, mosaic


## YOL0v5 (You Only Look Once)
For object detection model, state of the art Yolov5 was selected. Yolov5 is a single stage object detector. This means that there is a fast inference time giving that the region proposal and classification is done in in step.

### Metric 
mAP@.5:.95 as the evaluation metric 

## Results
From the 4 runs, it showed that model with bigger image input size and applying augmentation improved the final mAP result.

|        Models        | mAP_0.5 | mAP_0.5:0.95 |
|:--------------------:|:-------:|:------------:|
| yolov5_640_w/augemnt |   0.96  |     0.75     |
| yolov5_320_w/augemnt |   0.94  |     0.70     |
|      yolov5_640      |   0.81  |     0.53     |
|      yolov5_320      |   0.75  |     0.44     |

![mAP_scores.png](assets/mAP_scores.png)

### Evaluation on Test Dataset

Using the trained weights, the model is evaluated on the test dataset. The performance is similar to the validation dataset.

|        Models        |   Dataset  | mAP@.5  | mAP@.5:.95 |
|:--------------------:|:----------:|:-------:|:----------:|
| yolov5_640_w/augemnt | validation |   0.96  |    0.752   |
| yolov5_640_w/augemnt |    Test    |  0.977  |    0.781   |

### Inference on Test Dataset

The bounding boxes are generally well fitted and aligned with the ground truth (original annotation)

![inference.png](assets/inference.png)

### Inference on CNA News Clips

Using the trained model, it can also be used on video format. Below is a .gif converted from news video clip from CNA youtube channel.

![inference_on_news.gif](assets/inference_on_news.gif)

