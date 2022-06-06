import os

import pandas as pd

root_dir = '/Users/rosana.eljurdi/PycharmProjects/SegVal_Project/Results/csv'


for _,_,dirs in os.walk(root_dir):
    for dir in dirs:
        print(dir)
        f1 = pd.read_csv(os.path.join(root_dir, dir))[' dice'].mean()
        print(dir, f1)