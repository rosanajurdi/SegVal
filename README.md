# SegVal_Project
This is a directory that contains the code used to realize the paper: ... on validation of Segmentation models.

# Basecode, Installation and Dependencies  
The code for training the models is an extension of https://github.com/LIVIAETS/boundary-loss. \
For installation and dependencies, please check this [repository](https://github.com/LIVIAETS/boundary-loss). \
In addition to the requirements in [repository](https://github.com/LIVIAETS/boundary-loss), you are also required to install : \
The repository is composed of 4 sub-directories: 
- Data_Preprocessing : contains functions and modalities that allow you to split your datasets into folds, convert them to npy, and preprocess them for your code.
- Training: contains the scripts neccessary to train your models on the considered datasets and perform inference. This is an extension of the very well known [repository](https://github.com/LIVIAETS/boundary-loss). 
- Results: contains scripts that will allow you to perform 3D segmentation evaluation on your models.
- Confidence_Intervals: generates the confidence intervals over the adopted metrics.


# 
