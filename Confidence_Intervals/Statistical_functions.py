from re import A
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
from datetime import date

def statistical_analysis(df):
    "function that getd as input a pandas dataframe and produced some statistical values:"
    mean = np.round(df.mean().values[0]*100, 2)
    std = np.round(df.std().values[0]*100, 2)
    SEM = np.round(std / np.sqrt(len(df)), 2)
    w = 2*1.96*SEM
    print(str(mean) + "&" + str(std) + "&" + str(SEM) + "&" + str(w))

def create_subsampling_data(K_samples, data, save_dir):
    """
     # accumulate 100 statistical values on a subset of size K : {10, 20, 30, 50, 100}
     and stores them in a csv file of columns: sample-set	u-kj	sigma-kj	SEM-kj
     returns the absolute path to the csv file.

    """
   
    SET_k = []
    SET_u_k_j = []
    SET_sigma_k_j = []
    SET_SEM_k_j = []
    for k in K_samples:
        # SET_u_k_j set of means of the sample set of size k < N=110 for hippocampus 

        for j in tqdm(range(0,100)): # the experiment needs to be repeated a 100 times since in small samples the values will highly depend on sampled data.
            dice_acccuracies_Skj= data['metric'].sample(k).reset_index(drop=True) #N samples were selected without replacement

            # the mean of the sub samples 
            dice_mean = np.mean(dice_acccuracies_Skj)# get the mean of the sample of size k
            dice_std = np.std(dice_acccuracies_Skj) # get the standard deviation of the sample of size k 

            SET_k.append(k) 
            SET_u_k_j.append(dice_mean) # list of means of 100 k samples 
            SET_sigma_k_j.append(dice_std)  # list of standard deviation of 100 k samples 
            SET_SEM_k_j.append(dice_std/np.sqrt(k)) # list of SEM of a 100 k samples

    sample_data = {'sample-set':SET_k, 'u-kj': SET_u_k_j,'sigma-kj': SET_sigma_k_j, 'SEM-kj': SET_SEM_k_j}
    df = pd.DataFrame(sample_data)
    today = date.today()
    d4 = today.strftime("%b%d")
    print(d4)
    file = "subsampled-stats-{}-{}.csv".format("Hippo",str(d4))
    df.to_csv(os.path.join(save_dir,file))

    return os.path.join(save_dir,file)

