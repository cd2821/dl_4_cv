# DL for CV Final Project
## Author: Carmem Domingues 
## UNI: cd2821


The main code of this repo is contained under dl_4_cv. That's there I developed my code, with the other folders containing data used in the training or prediction steps. The main file is thus at dl_4_cv/cd2821_finalproject.ipynb. 

The notebook is organized in 2 main parts:
1. Image Classification
2. Object Detection

Each of the 2 parts above has 3 models, labeled 1-3, in the code in the markdown sections, with their training and results as relevant in order. 

For more details on each model, please see the Final Report and the Final Presentation, both submitted through Gradescope and copied at the root of this zip for completion. 

For more details of the most relevant files in each folder, and its usage, see below.

Thanks! =)

This repo is organized as follows (by folder):

### caffe
A clone of the caffe repo at (https://github.com/weiliu89/caffe/tree/ssd), used to get the labelmap for the VGG16 SSD model trained on ImageNet (Model 3 of the Object Detection part)
#### Relevant files: 
Labels for model: /data/ILSVRC2016/labelmap_ilsvrc_det.prototxt

### caffe_VGGNetSSD_ILSVR2016_models
A download of the SSD VGG16 pre-trained model (used by Model 3 of the OD part)
#### Relevant files: 
caffe model: VGGNet/ILSVRC2016/SSD_300x300/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel
proto file of model: VGGNet/ILSVRC2016/SSD_300x300/deploy.prototxt'

### deep-learning-opencv
Code from (https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/) used to get pre-trained model for Model 2 of the OD part
#### Relevant files: 
proto file of model: bvlc_googlenet.prototxt
caffe model: bvlc_googlenet.caffemodel
labels file: synset_words.txt'

### dl_4_cv
Carmem's repo, where the main code lives (cd2821_finalproject.ipynb)
#### Relevant files: 
history of model training from classification (informational only): history/*
logs of model training from classification (informational only): logs/*
model files for model training from classification (informational only): models/*
main file containing all code for project (notebook): cd2821_finalproject.ipynb
main file containing all code for project (.py version, saved as .py from notebook): cd2821_finalproject.py

### Images
Where images are stored, split by train/validation and plastic/noplastic classes
#### Relevant files: 
training data (plastic): data_for_modeling/train/plastic
training data (noplastic): data_for_modeling/train/noplastic
validation data (plastic): data_for_modeling/validation/plastic
validation data (noplastic): data_for_modeling/validation/noplastic


### object-detection-deep-learning
Code from (https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/#comment-458110) used to get pre-trained model for Model 1 of the OD part
#### Relevant files: 
caffe model: MobileNetSSD_deploy.caffemodel
proto file of model: MobileNetSSD_deploy.prototxt.txt


### weights
Folder containing weights file for pre-trained VGG16 used in the classification part, models 2 and 3
#### Relevant files: 
file of weights: vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5


