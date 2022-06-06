import os
import matplotlib.pyplot as plt
from PIL import Image
gt_path = '/Users/rosana.eljurdi/Desktop/fisheye/gtLabels'
val_path = '/Users/rosana.eljurdi/Desktop/fisheye/best_model/val'
baseline_path = '/Users/rosana.eljurdi/Desktop/fisheye/baseline/val'
img_path = '/Users/rosana.eljurdi/Desktop/fisheye/rgb_images'
import numpy as np
#dictt = {'191': , '255': ,'64':, '32':   }
for gts in os.listdir(gt_path):
    if '.png' in gts:
       #plt.imsave(os.path.join(gt_path, gts), np.array(Image.open(os.path.join(gt_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(gt_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(baseline_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(val_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(img_path, gts))))


    '''
    if len(np.unique(a)) >8 :


        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(gt_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(baseline_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(val_path, gts))))
        plt.figure()
        plt.imshow(np.array(Image.open(os.path.join(img_path, gts))))
    '''


    print("done")
