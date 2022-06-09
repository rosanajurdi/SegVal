'''

script to test the validity of the results and to which folds they belong to.
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
from utils import dice_coef, class2one_hot, dice_acc_3D, haussdorf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

prediction_dir = '/Results/Hippo/predictions_npy'
gt_dir = '/Users/rosana.eljurdi/Documents/Confidence_Intervals_Olivier/Task04_Hippocampus/Splits/test/test_npy/gt_npy'

df_dice = pd.DataFrame()
hauss = pd.DataFrame()
cerror = pd.DataFrame()

name = []
dice_array = []
path = Path(prediction_dir)
patient_ids: List[Path] = list(path.rglob('*.npy'))
import torch
dices = []
for patient in patient_ids:
    path = Path(patient)
    name = path.name
    Volume_3D_pred = torch.tensor(np.load(path).argmax(axis=1)[0]).reshape(1,256,256)
    Volume_3D_gt = torch.tensor(np.load(os.path.join(gt_dir, name))).reshape(1,256,256)
    dices.append(dice_acc_3D(Volume_3D_pred, Volume_3D_gt))
    print(np.mean(dices))
