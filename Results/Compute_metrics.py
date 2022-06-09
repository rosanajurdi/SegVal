'''
@author: eljurros
'''
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
df_all = []
from pathlib import Path
import os
import sys
sys.path.append('/Users/rosana.eljurdi/PycharmProjects/SegVal_Project/Training')
from typing import List
from utils import dice_coef, class2one_hot, dice_acc_3D, haussdorf, hausdorff_deepmind

import torch
import nibabel as nib

import numpy as np
from matplotlib import pyplot as plt
'''
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
data2D = np.random.random((50, 50))
im = plt.imshow(data2D, cmap="copper_r")
plt.colorbar(im)
plt.show()
'''

#root_dir = '/Users/rosana.eljurdi/Documents/Confidence_Intervals_Olivier/Task04_Hippocampus/Splits/test/test_npy/results/fold_1/Patient_RESULTS'
#prediction_dir = os.path.join(root_dir, 'predictions_npy' )
#gt_dir = os.path.join(root_dir, 'gt_npy' )

prediction_dir = '/Results/Hippo/predictions_npy'
gt_dir = '/Users/rosana.eljurdi/Documents/Confidence_Intervals_Olivier/Task04_Hippocampus/Splits/test/test_npy/gt_npy'


df_dice = pd.DataFrame()
hauss = pd.DataFrame()
cerror = pd.DataFrame()

name = []
dice_array = []


def Get_patients(root_dir):
    name_list = []
    for id_file in os.listdir(root_dir):
        try:
            patient_name = '{}_{}'.format(id_file.split('_')[0], id_file.split('_')[1])
            if patient_name not in name_list:
                name_list.append(patient_name)
        except:
            pass
    return name_list
def Get_Mean_Scores(df):
    assert ( df[' dice'].size == df[' haussdorf'].size == df['connecterror '].size == df['file'].size )
    dice= np.mean(df[' dice'])
    hauss = np.mean(df[' haussdorf'])
    print([dice, hauss])
    return [dice, hauss]
    print('passed')

ds_length = {'hippocampus': 110, 'brain': 324}
ds_class = {'hippocampus': 3, 'brain': 4}

def fix_it(slice):
    a = np.array(torch.tensor(slice.argmax(axis=0)))
    return a
patient_ids = Get_patients(prediction_dir)
key = patient_ids[0].split('_')[0]
#assert ds_length[key] == len(patient_ids)  # assertin command to make sure that the patient numbers are conserved

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
'''

script that reads the npy results and generates the result.csv files wirh patient dice accuracies.
'''

fold_all_H1 = open('/Results/Hippo/csv/results-dice.csv', "w")
fold_all_H1.write(f"file, metric \n")

fold_all_H2 = open('/Results/Hippo/csv/results-hauss.csv', "w")
fold_all_H2.write(f"file,metric\n")
for patient in patient_ids:
    path = Path(prediction_dir)
    pred_paths: List[Path] = list(path.rglob('{}_*'.format(patient)))
    basenames = [p.name for p in pred_paths]
    npy_stacked = [np.load(p) for p in pred_paths]
    Volume_3D_pred = [fix_it(np.load(p)[0]) for p in pred_paths]
    '''
    for slice, file in zip(*[Volume_3D_pred, basenames]):
        plt.imsave(os.path.join('../Results/predictions_img','{}.png'.format(file.split('.npy')[0]) ), slice )
    '''
    gt_paths: List[Path] = list(path.rglob('{}_*'.format(patient)))
    Volume_3D_gt = [np.load(os.path.join(gt_dir, p)) for p in basenames]
    '''
    for slice, file in zip(*[Volume_3D_gt, basenames]):
        plt.imsave(os.path.join('../Results/gt_img','{}.png'.format(file.split('.npy')[0])), slice)
    '''

    dice = dice_acc_3D(np.array(Volume_3D_pred),
                       np.array(Volume_3D_gt))

    fold_all_H1.write(f"{patient}, {np.float(dice.mean())} \n")

    hd = hausdorff_deepmind(np.array(Volume_3D_pred),
                  np.array(Volume_3D_gt))

    fold_all_H2.write(f"{patient}, {np.float(hd.mean())} \n")
    print(patient, dice.mean(), hd.mean())

    #hauss = haussdorf(torch.tensor(np.array(Volume_3D_pred)),
    #                   torch.tensor(np.array(Volume_3D_gt)))

