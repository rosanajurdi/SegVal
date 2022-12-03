'''
@author: eljurros

Note that the current deep mind implementation of the haussdorf loss behaves as following:
If one of the masks is empty, the corresponding lists are empty and all distances in
  the other list are `inf`.

For the time being I will discard the inf until I find a suitable solution.
'''
import pandas as pd
df_all = []
from pathlib import Path
import os
import sys
from typing import List

from surface_distance import metrics
import torch


#prediction and ground-truth directories:
root_path = '/Users/rosana.eljurdi/Documents/Projects/Conf_Seg/Confidence_Intervals_Olivier/Task04_Hippocampus/Splits/test/test_npy'
fold_path = os.path.join(root_path, '')
#prediction and ground-truth directories:
prediction_dir = os.path.join(fold_path,'fold_3/predictions_npy')
gt_dir = os.path.join(root_path,'gt_npy')

fold_all_H1 = open(os.path.join(fold_path,'results-Dice-3D-L1.csv'), "w")
fold_all_H1.write(f"file, metric \n")

fold_all_H2 = open(os.path.join(fold_path,'results-hauss-3D-L1.csv'), "w")
fold_all_H2.write(f"file,metric\n")

df_dice = pd.DataFrame()
hauss = pd.DataFrame()

name = []
dice_array = []


def Get_patients(root_dir):
    "returns list of all patients in the dataset or 3D examples"
    name_list = []
    for id_file in os.listdir(root_dir):
        try:
            patient_name = '{}_{}'.format(id_file.split('_')[0], id_file.split('_')[1])
            if patient_name not in name_list:
                name_list.append(patient_name)
        except:
            pass
    return name_list


ds_length = {'hippocampus': 110, 'brain': 324}
# ds_class : classes + background


def fix_it(slice):
    '''

    :param slice: takes a slice image in one hotspot form
    :return: returns the class maps in tznsor format
    '''
    a = np.array(torch.tensor(slice.argmax(axis=0)))
    return a

# here is where the code begins
patient_ids = Get_patients(prediction_dir)
key = patient_ids[0].split('_')[0]
#assert ds_length[key] == len(patient_ids)  # assertin command to make sure that the patient numbers are conserved

import numpy as np

'''

script that reads the npy results and generates the result.csv files wirh patient dice accuracies.
'''

assert len(np.unique(patient_ids)) ==  ds_length[key]
for patient in patient_ids:
    path = Path(prediction_dir)
    pred_paths: List[Path] = list(path.rglob('{}_*'.format(patient)))
    basenames = [p.name for p in pred_paths]

    Volume_3D_pred_multi = [fix_it(np.load(p)[0]) for p in pred_paths]
    Volume_3D_pred = np.array([np.where(a == 2, 0, a) for a in Volume_3D_pred_multi]) # Choosing the first region L1
    assert len(np.unique(Volume_3D_pred)) == 2
    Volume_3D_pred = Volume_3D_pred.astype(bool) # transforming to boolean

    '''
    for slice, file in zip(*[Volume_3D_pred, basenames]):
        plt.imsave(os.path.join('../Results/predictions_img','{}.png'.format(file.split('.npy')[0]) ), slice )
    '''
    gt_paths: List[Path] = list(path.rglob('{}_*'.format(patient)))



    Volume_3D_gt_multi = [np.load(os.path.join(gt_dir, p)) for p in basenames]
    Volume_3D_gt = np.array([np.where(a == 2, 0, a) for a in Volume_3D_gt_multi])  # Choosing the first region L1
    Volume_3D_gt = Volume_3D_gt.astype(bool) # transforming to boolean

    #dice = dice_acc_3D(np.array(Volume_3D_pred),
    #                   np.array(Volume_3D_gt), 2)

    dice = np.round(metrics.compute_dice_coefficient(Volume_3D_gt, Volume_3D_pred)*100,2)

    fold_all_H1.write(f"{patient}, {np.round(dice, 2)} \n")

    sf = metrics.compute_surface_distances(np.array(Volume_3D_pred),
                 np.array(Volume_3D_gt), [1,1, 1]) # 1mm^^ corresponding to  mm spacing

    hd = metrics.compute_robust_hausdorff(sf, 95)

    fold_all_H2.write(f"{patient}, {np.round(hd, 2)} \n")
    print(patient, np.round(dice, 2), np.round(hd, 2))



