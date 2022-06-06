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


# Data_Preprocessing: 
This directory is mainly dedicated to two functionalities: 1) splitting the datasets to folds and 2) converting the data to numpy and preprocessing them. 

## splitting the datasets to folds
## converting the data to numpy
```
'--source_dir': the directory to find the nifty data                  
'--dest_dir': the directory to store the npy data                 
'--shape': reshaoing the data by defualt is  default=[256, 256]
'--retain': "Number of retained patient for the validation data in our code it is set to default=0 since we are pre-identifying the folds in a seperate script"
'--type': the type of folds you are constructing could be "val or test", in case val it converts both train and validation folders, in case test it ony cnverts the test file to numpy 
'--discard_negatives' set to  default=False since we want all the slices in the dataset empty and full. 
```

