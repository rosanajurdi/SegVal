'''
Created on Feb 13, 2020
@author: eljurros
'''
'''
Created on Mar 20, 2019
@author: Rosana EL JURDI: 
Script to divide the data into folds,
1) Import the required dataset class from DataSEt_Classes and initialize an instance of the class ds.
2) root_path denotes the absolute path to the dataset + ROOT 
3) fold: the fold you need to create.
4)
# variables to initialize
typ = 'ROOT': the root folder you want your downloaded dataset to be in. preferably place it in ROOT. 
root_path: the root directory leading to your data.
fold: the name of the target fold 
nb_val: the number of validation samples (recommended to be 20% of the total training set)
'''
import os
import random
import shutil

from DataSEt_Classes import Decathlon

typ = 'ROOT'
root_path = '/Users/rosana.eljurdi/Desktop/Benchmark/Task02_Heart/nifty'
fold = 'FOLD_DUmmy'
nb_val = 400

if os.path.exists(os.path.join(root_path, fold)) is False:
    os.mkdir(os.path.join(root_path, fold))
    os.mkdir(os.path.join(root_path, fold, 'train'))
    os.mkdir(os.path.join(root_path, fold, 'val'))
inner_arr = []
outer_arr = []
ds = Decathlon(root_dir=root_path, typ=typ)
train_path = [os.path.join(root_path, fold, 'train', 'imagesTr'), os.path.join(root_path, fold, 'train', 'labelsTr')]
val_path = [os.path.join(root_path, fold, 'val', 'imagesTr'), os.path.join(root_path, fold, 'val', 'labelsTr')]
if os.path.exists(train_path[0]) is False:
    os.mkdir(train_path[0])
    os.mkdir(train_path[1])

if os.path.exists(val_path[0]) is False:
    os.mkdir(val_path[0])
    os.mkdir(val_path[1])

for i, patient_path in enumerate(ds.filename_pairs):
    patient_name = os.path.basename(patient_path[0])
    input_filename, gt_filename = patient_path[0], \
                                  patient_path[1]

    i = random.randint(1, 101)

    if i < 50 and nb_val > 0:
        nb_val = nb_val - 1

        shutil.copy(patient_path[0], val_path[0])
        shutil.copy(patient_path[1], val_path[1])
        with open(os.path.join(os.path.join(root_path, fold, 'val.txt')), 'a') as the_file:
            the_file.write(patient_name)
            the_file.write('\n')

    else:
        shutil.copy(patient_path[0], train_path[0])
        shutil.copy(patient_path[1], train_path[1])
        with open(os.path.join(os.path.join(root_path, fold, 'train.txt')), 'a') as the_file:
            the_file.write(patient_name)
            the_file.write('\n')