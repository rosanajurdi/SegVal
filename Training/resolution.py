import nibabel as nb
import numpy as np
import os
x =[]
y= []
z = []
path = "/Users/rosana.eljurdi/Documents/Desktop/Task04_Hippocampus/imagesTr"
for patient in os.listdir(path):
    D3_volume = nb.load(os.path.join(path, patient))
    print(D3_volume.header.get_dim_info())
    x.append(D3_volume.shape[0])
    y.append(D3_volume.shape[1])
    z.append(D3_volume.shape[2])

print(np.min(x), np.max(x))
print(np.min(y), np.max(y))
print(np.min(z), np.max(z))
