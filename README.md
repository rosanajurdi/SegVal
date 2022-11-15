# SegVal_Project
This is a directory that contains the code used to realize the paper:  ["HOW PRECISE ARE PERFORMANCE ESTIMATES FOR TYPICAL MEDICAL IMAGE
SEGMENTATION TASKS?"](https://github.com/rosanajurdi/SegVal/edit/master). The directory is mainly composed of 3 parts: S1 - The data pre-processing part, 
S2- the training part ( which is the state-of-the-art code found in [repository](https://github.com/LIVIAETS/boundary-loss) and S3) the Confidence interval package 
resposible for generating the statistical evaluation and tables in the paper. 

# Basecode, Installation and Dependencies  
The code for training the models is an extension of https://github.com/LIVIAETS/boundary-loss. \
For installation and dependencies, please check this [repository](https://github.com/LIVIAETS/boundary-loss). \
In addition to the requirements in [repository](https://github.com/LIVIAETS/boundary-loss), you are also required to install : \
The repository is composed of 4 sub-directories: 
- Data_Preprocessing : contains functions and modalities that allow you to split your datasets into folds, convert them to npy, and preprocess them for your code.
- Training: contains the scripts neccessary to train your models on the considered datasets and perform inference. This is an extension of the very well known [repository](https://github.com/LIVIAETS/boundary-loss). 
- Results: contains scripts that will allow you to perform 3D segmentation evaluation on your models.
- Confidence_Intervals: generates the confidence intervals over the adopted metrics.


# Preprocessing
The data was split into 100 samples for training, 50 for validation and the rest for testing. The training, testing and validation patients can be found
in the following text files: [train file](), [test file](), and [val file](). Note that the data structure needs to be of the following format. 

```
Directory-path-to-data/
    ├── root
    │   └── train
    │           ├── imagesTr
    │           └── labelsTr
    │   └── val
    │           ├── imagesTr
    │           └── labelsTr
    │       
    └──  test
    │       └── imagesTr
    │           ├── hippocampus_id.nii.gz
    │           └── hippocampus_id.nii.gz
    │       └── labelsTr
    │           ├── hippocampus_id.nii.gz
    │           └── hippocampus_id.nii.gz

```
The script assumes that the splits of the data into train, test and val is already done. It will produce an error if the paths to the 
data was false. The script needs to be run twice: once with --args.type == val and a second time with --args.type == 'test'. 
Note that val needs to be before test else the code will throw an assertion error.  There are two scripts one for the Hipppocampus dataset
[slice_decathlon]() and [slice_braintumor]().

```
'--source_dir': the directory to find the nifty data having train, test and val                
'--dest_dir': the directory to store the npy data :  needs to not exist in first run under --type = val.                  
'--shape': reshaping the data by defualt is  default=[256, 256]
'--retain': "Number of retained patient for the validation data in our code it is set to default=0 since we are pre-identifying the folds in a seperate script"
'--type': the type of folds you are constructing could be "val or test", in case val it converts both train and validation folders, in case test it ony cnverts the test file to numpy 
'--discard_negatives' set to  default=False since we want all the slices in the dataset empty and full. 
```

# Training: 
## main training script for the dataset: 

Replica of [](), please check the code documentation for further information. 
## Inference: 
please check the code documentation for further information.

## Computing Metrics:  
The script for the ISBI Paper on the whole hippocampus is [Compute_metric_3D_WholeHippo]().
The input to this script is mainly the path to the 2D gt_npy and prediction directories. The script gathers the 2D slices in both gt and 
prediction directories and computes the 3D performance metrics : Dice and 95% hausdorff. In the ISBI paper, the hippocampus is considered 
as a whole entity, therefore, both classes are merged into one and the PMs are computed for the whole hippocampus. 

You can also check [Compute_metric_3D]() to calculate the PMs of each region independently. Please check the code documentation for
further information.

# Confidence Intervals: 

To compute the confidence intervals, you need to check the following [sub directory]() and the [main.py](). The code here is 
fully monotone as you only need to specify some initial parameters at the begining and all the tables are then generated. 
The code computes the 3D average PM in table 1 of the paper; the corresponding Gaussian values in table 3 both analytical and 
bootstrapping values as in table 2. The code saves these values in txt files in a format compatible with overleaf so you 
can directly copy and paste it into your overleaf project. 

The code will also generate the sub-samples csv file which contains the analytical and bootstrap values depending on 
parameter k. All these files are saved in the [Stats]() directory under a sub-directory having the following name format:
```
subsampled-stats-{PM}-{datasetname}-MonthDateTime
```
The Stats (result) directory has the following format:

```
Stats/
    ├── Haussdorf-Distance
    │   └── subsampled-stats-hippo-HD-Nov1705
    │           ├── Boot_txt_log.txt
    │           └── subsampled-stats-Brain-Oct05.csv
    │           └── Boot_txt_log.txt
    │       
    └──  Dice_Accuracy
    │       └── subsampled-stats-hippo-Apr1010
    │            ├── Boot_txt_log.txt
    │           └── subsampled-stats-Brain-Oct05.csv
    │           └── Boot_txt_log.txt


```
